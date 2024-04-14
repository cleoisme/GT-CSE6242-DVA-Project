import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import BDay
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import dask.dataframe as dd
import dask
from dask.diagnostics import ProgressBar
from dask.delayed import delayed

from signet.cluster import Cluster
from scipy.sparse import csc_matrix

from sklearn.cluster import OPTICS, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import yfinance as yf
from scipy.linalg import eigh

import warnings

# set random state
np.random.seed(49)

# these are the static dtypes for the columns in the dataframe
dtypes = {'marketcap': 'float32',
          'close': 'float32',
          'volume': 'float32',
          'closeunadj': 'float32',
          'dollar_volume': 'float32',
          'marketcap_med3m': 'float32',
          'dollar_volume_med3m': 'float32',
          'closeunadj_med3m': 'float32'}

# 'top600mktcap.pkl'
def load_pkl_file(filename="liquidity_master.pkl"):
    for root, dirs, files in os.walk('C:/'):
        for file in files:
            if file == filename:
                filepath = os.path.join(root, file)
                df = pd.read_pickle(filepath)
                df = df.astype(dtypes)
                df.sort_index(inplace=True)
                return df
    raise FileNotFoundError(f"File {filename} not found in C:/")


def evaluate_singleP(df):
    df_new = df.loc[df.index.get_level_values('date')>="2022-04-05"]['close'].unstack().dropna(axis=1)
    df_new = df_new.pct_change().dropna()
    cluster_sponge = clusterSPONGE(df_new, k=30)
    return cluster_sponge, df_new


def generate_sample_corr_mat(df, n_dfs=5, residual_returns=True, look_back=60):
    # for testing this generates multiple correlation matrice to observe
    df = df['close'].unstack().sort_index()
    dates = df.index[look_back:]
    out = dict()

    for i in range(n_dfs):
        _date = np.random.choice(dates)
        _df = df.loc[_date - BDay(look_back):_date].dropna(axis=1)
        log_returns = np.log(_df/_df.shift(1)).dropna()

        if residual_returns:
            mkt_data = pd.read_pickle('spx.pkl')[['log_returns']]
            mkt = mkt_data.loc[log_returns.index].fillna(method='ffill')
            reg = LinearRegression()
            reg.fit(mkt.values.reshape(-1, 1), log_returns.values)
            resid_returns = log_returns - reg.predict(mkt.values.reshape(-1, 1))
            out[str(i)] = resid_returns.corr()
        else:
            out[str(i)] = log_returns.corr()
    return out


def optimal_number_of_clusters(data, q):
    """
    Determine the optimal number of clusters using Random Matrix Theory and the Marchenko-Pastur distribution.

    Parameters:
    - data: A (n_samples, n_features) array of data points.
    - q: The ratio of the number of features to the number of samples (n_features / n_samples).

    Returns:
    - int: Optimal number of clusters.
    """
    # Step 1: Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)

    # Step 2: Compute eigenvalues from the correlation matrix
    eigenvalues, _ = eigh(corr_matrix)

    # Step 3: Calculate the Marchenko-Pastur maximum eigenvalue
    lambda_plus = (1 + np.sqrt(1/q))**2

    # Step 4: Count the number of eigenvalues above the upper Marchenko-Pastur bound
    significant_eigenvalues = eigenvalues[eigenvalues > lambda_plus]

    return len(significant_eigenvalues)


def test_conseq_clusters(df, date1="2023-03-14", date2="2023-03-15", look_back=60, k=20):
    df = df['close'].unstack().sort_index()
    d1 = compute_residuals_for_date(pd.to_datetime(date1), df, lookback_window=look_back, k=k)
    d2 = compute_residuals_for_date(pd.to_datetime(date2), df, lookback_window=look_back, k=k)

    both = pd.concat([d1, d2])
    turnover = get_cluster_turnover(both.reset_index())
    return turnover
   

def generate_sorted_corr_matx(corr, labels):
    plt.clf()
    labels = np.argsort(labels)
    sorted_corr = corr.iloc[labels, labels]

    sns.heatmap(sorted_corr)
    plt.show()


def make_pos_neg(corr):
    # preps the correlation matrix to be inputed into SPONG
    pos, neg = corr.copy().values, corr.copy().values
    pos[pos<0]=0
    neg[neg>0]=0
    neg = np.abs(neg)
    return csc_matrix(pos), csc_matrix(neg)


def run_eval(k=20):
    df = load_pkl_file()
    df = scrub_df(df)

    corrs = generate_sample_corr_mat(df)
    for i in corrs.keys():
        corr1 = corrs[i]
        pos, neg = make_pos_neg(corr1)
        c = Cluster((pos, neg))
        labels = c.SPONGE(k=k)
        generate_sorted_corr_matx(corr1, labels)


def resample_multidf_monthly_wffill(df, col_batch_size=5, limit=5, freq='W'):
    # copy and remove duplicates
    df = df.copy(deep=True)
    df = df.loc[~df.index.duplicated(keep='first')]

    # keeps the same data-types to conserve memory
    dtype_dict = dict(df.dtypes)

    dfs = []
    for i in range(0, len(df.columns), col_batch_size):
        print(i)
        _df = df.iloc[:, i:i + col_batch_size].unstack()
        _df = _df.asfreq('d').ffill(limit=limit).asfreq(freq).stack()
        dfs.append(_df)

    # join back all the columns
    df = pd.DataFrame().join(dfs, how='outer')
    df = df.astype(dtype_dict)
    return df


def test_turnover_for_cluster(df, cluster_id, date1, date2):
    # Filter the DataFrame for the specific cluster and the two dates
    cluster_data = df[(df['cluster'] == cluster_id) & (df['date'].isin([date1, date2]))]
    
    # Get the sets of tickers for each of the two dates
    tickers_date1 = set(cluster_data[df['date'] == date1]['ticker'])
    tickers_date2 = set(cluster_data[df['date'] == date2]['ticker'])
    
    # Calculate the turnover
    total_tickers = tickers_date1.union(tickers_date2)
    changed_tickers = tickers_date1.symmetric_difference(tickers_date2)
    turnover = len(changed_tickers) / len(total_tickers) if total_tickers else 0
    
    # Print the results
    print(f"Cluster {cluster_id} turnover from {date1} to {date2}: {turnover:.2%}")
    print(f"Tickers on {date1}: {tickers_date1}")
    print(f"Tickers on {date2}: {tickers_date2}")
    print(f"New tickers on {date2}: {tickers_date2 - tickers_date1}")
    print(f"Removed tickers on {date2}: {tickers_date1 - tickers_date2}")


def calculate_turnover(set_day1, set_day2):
    if not set_day1 or not set_day2:
        return None
    total_tickers = set_day1.union(set_day2)
    changed_tickers = set_day1.symmetric_difference(set_day2)
    return len(changed_tickers) / len(total_tickers)


def get_cluster_turnover(df):
    pivot_df = (df.reset_index().set_index(['cluster', 'date'])['ticker']
               .groupby(level=[0, 1])
               .apply(set)
               .unstack('date'))
    
    # Calculate the turnover for each cluster across consecutive days
    turnovers = pivot_df.apply(lambda row: [calculate_turnover(row[i], row[i+1]) for i in range(len(row)-1)], axis=1)

    # Create a new DataFrame from the turnovers data
    turnover_df = pd.DataFrame(turnovers.tolist(), index=pivot_df.index, columns=pivot_df.columns[1:])

    # Reset index to turn 'cluster' back into a column
    turnover_df = turnover_df.reset_index()

    # Melt the DataFrame to have one row per cluster per date
    turnover_df = turnover_df.melt(id_vars='cluster', var_name='date', value_name='turnover')

    # Convert turnover to percentage
    turnover_df['turnover'] = turnover_df['turnover'] * 100
    return turnover_df
    

def scrub_df(df):
    # can add any logic here to scrub the df, for now set to in_universe=True
    # marketcap_med3m > 50
    # dollar_volume_med3m > 250_000
    # closeunadj_med3m > 5

     # marketcap_med12m > 350
    # dollar_volume_med12m > 500000
    # closeunadj_med3m > 5

    # NOTE - I want to make a slower moving univserse with less stocks
    # NOTE - goal is to cut down on cluster assignment turnover
    # df['marketcap_med12m'] = df.groupby(by='ticker')['marketcap'].rolling(252).median().droplevel(0)
    # df['dollar_volume_med12'] = df.groupby(by='ticker')['dollar_volume'].rolling(252).median().droplevel(0)
    # df['closeunadj_med12m'] = df.groupby(by='ticker')['closeunadj'].rolling(252).median().droplevel(0)
    df['marketcapQ'] = df.groupby(by='date')['marketcap'].apply(lambda x: pd.qcut(x, q=5, labels=False)).droplevel(0)

    # take only the top quintile of highest market cap companies
    df = df.loc[df.marketcapQ==4]
    return df


def find_cum_var(symetric_matrix, var_explained=.80):
    eigenvalues, _ = np.linalg.eigh(symetric_matrix)

    # Sort the eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate the total variance
    total_variance = np.sum(eigenvalues)

    # Calculate the cumulative sum of eigenvalues
    cumulative_variance = np.cumsum(eigenvalues)

    # Calculate the percentage of explained variance
    explained_variance_ratio = cumulative_variance / total_variance

    # Find the number of eigenvalues needed to explain 95% of the variance
    num_eigenvalues = np.argmax(explained_variance_ratio >= var_explained) + 1
    return num_eigenvalues


def tuneOpticsPCS(data):
    best_score = -1
    best_params = {}

    for min_samples in tqdm(range(2, 10)):  # Example range, adjust based on your data size
        for max_eps in tqdm(np.arange(0.5, 2.0, 0.5)):  # Example range, adjust based on the data scale
            optics_model = OPTICS(min_samples=min_samples, max_eps=max_eps, n_jobs=8)
            labels = optics_model.fit_predict(data)
            
            # Evaluate only clusters with more than one element
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'min_samples': min_samples, 'max_eps': max_eps}
                    print(f'Best Params: {best_params} | {score}')
    return best_params


def clusterOpticsPCS(past_returns, n_components=.80):
    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components))])
    pipe.fit(past_returns)
    pcs = pipe.steps[1][1].components_.T
 
    # fit the optics algorithm
    best_params = tuneOpticsPCS(pcs)
    cluster = OPTICS(**best_params)
    print(f'fitting cluster algos on shape ... {pcs.shape}')
    cluster.fit(pcs)
    labels = cluster.labels_
    out = pd.DataFrame(pcs, index=past_returns.columns, columns=[f'pc_{i}' for i in range(pcs.shape[1])])
    out['Opticslabels'] = labels
    return out


def clusterSPONGE(past_returns, k=30, mp=False):
    np.random.seed(49)
    if mp:
        k = optimal_number_of_clusters(past_returns, past_returns.shape[1]/past_returns.shape[0])
        print(f'Found Optimal K based on MP: {k}')
    corr_matx = past_returns.corr()
    
    # define the positive matrix
    corr_pos, corr_neg = corr_matx.copy().values, corr_matx.copy()
    corr_pos[corr_pos<0] = 0.
    corr_neg[corr_neg>0] = 0.

    # now convert the negative matrix to positive 
    corr_neg = np.abs(corr_neg)

    # needs to be in sparse COO format!
    corr_pos = csc_matrix(corr_pos, dtype='float32')
    corr_neg = csc_matrix(corr_neg, dtype='float32')

    # cluster
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c = Cluster((corr_pos, corr_neg))
        labels = c.SPONGE(k=k)
        return labels


def compute_residuals_for_date(date, df, lookback_window, k=20):
    # NOTE - should I use log returns here?
    # Grab the stocks governed by the look-back period and compute percent changes
    window = df.loc[date - BDay(lookback_window):date].dropna(axis=1)
    log_returns = np.log(window/window.shift(1)).dropna()
    raw_returns = window.pct_change().dropna()

    # take the values from t-1 through t-look-back, these are the in-sample values
    # save t0-t1 as the out of sample residual return to append and send back out
    mkt_data = pd.read_pickle('spx.pkl')[['log_returns']]
    mkt_return = mkt_data.loc[log_returns.index].fillna(method='ffill').values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X=mkt_return[:-1], y=log_returns.iloc[:-1])

    # compute in-sample residual returns to fit the clustering algorithm on
    in_sample_pred = reg.predict(mkt_return[:-1])
    in_sample_resid = log_returns.iloc[:-1] - in_sample_pred

    # fit SPONGE Algorithm and compute the labels where k=30
    labels = clusterSPONGE(in_sample_resid, k=k)

    # Now predict the market impact and compute residual returns for the most recent date
    new_return = log_returns.iloc[-1:]
    residual = new_return - reg.predict(mkt_return[-1].reshape(1, -1))
    residual_df = pd.DataFrame(residual, index=[date], columns=new_return.columns)

    # store the residual returns
    residual_df = residual_df.stack().to_frame("residual_return")
    residual_df['cluster'] = labels
    dtypes = {'cluster': 'int8',
              'residual_return': 'float16',
              'raw_return': 'float16'}

    # return residual returns, actual returns and cluster assignments
    residual_df = residual_df.join(raw_returns.iloc[-1:].stack().to_frame("raw_return"))
    residual_df = residual_df.astype(dtypes)

    return residual_df


def residual_returns(df, lookback_window=252):
    # NOTE took my pc 87m 41s to complete
    # We use dask to avoid using pd.concat where we run out of memory
    df = df['close'].unstack().sort_index()
    dates = df.index[lookback_window:]

    # check to see if we have index data
    try:
        spy = pd.read_pickle('spx.pkl')
    except Exception as e:
        spy = yf.download("^GSPC", start="1990-01-01")
        spy['log_returns'] = np.log(spy['Adj Close']/spy['Adj Close'].shift(1))
        spy.to_pickle('spx.pkl')
   
    # Use dask.delayed to parallelize the computation
    delayed_results = [delayed(compute_residuals_for_date)(date, df, lookback_window) for date in tqdm(dates)]

    # Use Dask to manage the concatenation and computation
    dask_df = dd.from_delayed(delayed_results)

    with ProgressBar():
        out = dask_df.compute()

    return out


def main_data_prep(lookback=252):
    # load and scrub the data-frame
    df = load_pkl_file(filename='top600mktcap.pkl')
   
    # compute residual returns and concat all dates
    rr = residual_returns(df, lookback_window=lookback)
    rr = rr.reset_index().set_index(['date', 'ticker']).drop("level_0", axis=1)

    print("Saving Residual Returns to Pickle...")
    rr.to_pickle("residual_returns.pkl")


def run_example_algo():
    df = load_pkl_file(filename='top600mktcap.pkl')
    #df = scrub_df(df)

    corrs = generate_sample_corr_mat(df)
    for i in corrs.keys():
        corr1 = corrs[i]
        pos, neg = make_pos_neg(corr1)
        c = Cluster((pos, neg))
        labels = c.SPONGE(k=20)
        generate_sorted_corr_matx(corr1, labels)


def build_feat_df(df_path='residual_returns.pkl'):
    # cumulative returns
    rets = pd.read_pickle(df_path)
    rets = (1 + rets.groupby(by=['date', 'cluster'])['raw_return'].mean().unstack()).cumprod().stack().to_frame('ret')

    from utils.ml_utils import build_features
    feats = build_features(rets)
    
    print("Saving out features data....")
    feats.to_pickle("features.pkl")


def load_features_data():
    try:
        return pd.read_pickle("features.pkl")
    except Exception:
        build_feat_df()
        return pd.read_pickle("features.pkl")










   




   

        






   
    

