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

import warnings

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

    cluster_optics = clusterOpticsPCS(df_new, n_components=20)
    cluster_sponge = clusterSPONGE(df_new, k=30)

    return cluster_optics, cluster_sponge, df_new


def generate_sample_corr_mat(df, n_dfs=5, residual_returns=True):
    # for testing this generates multiple correlation matrice to observe
    df = df['close'].unstack().sort_index()
    dates = df.index[504:]
    out = dict()

    for i in range(n_dfs):
        _date = np.random.choice(dates)
        _df = df.loc[_date - BDay(504):_date].dropna(axis=1)
        log_returns = np.log(_df/_df.shift(1)).dropna()

        if residual_returns:
            mkt = log_returns.mean(axis=1)
            reg = LinearRegression()
            reg.fit(mkt.values.reshape(-1, 1), log_returns.values)
            resid_returns = log_returns - reg.predict(mkt.values.reshape(-1, 1))
            out[str(i)] = resid_returns.corr()
        else:
            out[str(i)] = log_returns.corr()
    return out

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


def scrub_df(df):
    # can add any logic here to scrub the df, for now set to in_universe=True
    return (df.loc[df.in_universe==True])


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


def clusterSPONGE(past_returns, k=30):
    np.random.seed(49)
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


def compute_residuals_for_date(date, df, lookback_window):
    # NOTE - should I use log returns here?
    # Grab the stocks governed by the look-back period and compute percent changes
    window = df.loc[date - BDay(lookback_window):date].dropna(axis=1)
    log_returns = np.log(window/window.shift(1)).dropna()
    raw_returns = window.pct_change().dropna()

    # take the values from t-1 through t-look-back, these are the in-sample values
    # save t0-t1 as the out of sample residual return to append and send back out
    mkt_return = log_returns.mean(axis=1).values.reshape(-1, 1) # this assumes an equal weight index
    reg = LinearRegression()
    reg.fit(X=mkt_return[:-1], y=log_returns.iloc[:-1])

    # compute in-sample residual returns to fit the clustering algorithm on
    in_sample_pred = reg.predict(mkt_return[:-1])
    in_sample_resid = log_returns.iloc[:-1] - in_sample_pred

    # fit SPONGE Algorithm and compute the labels where k=30
    labels = clusterSPONGE(in_sample_resid, k=30)

    # Now predict the market impact and compute residual returns for the most recent date
    new_return = log_returns.iloc[-1:]
    residual = new_return - reg.predict(mkt_return[-1].reshape(1, -1))
    residual_df = pd.DataFrame(residual, index=[date], columns=new_return.columns)

    # store the residual returns
    residual_df = residual_df.stack().to_frame("residual_return")
    residual_df['cluster'] = labels

    # we should also probably store the actual returns
    residual_df = residual_df.join(raw_returns.iloc[-1:].stack().to_frame("raw_return"))
   
    # Ensure the DataFrame contains all columns
    return residual_df


def residual_returns(df, lookback_window=252):
    # NOTE took my pc 87m 41s to complete
    # We use dask to avoid using pd.concat where we run out of memory
    df = df['close'].unstack().sort_index()
    dates = df.index[lookback_window:]
   
    # Use dask.delayed to parallelize the computation
    delayed_results = [delayed(compute_residuals_for_date)(date, df, lookback_window) for date in tqdm(dates)]

    # Use Dask to manage the concatenation and computation
    dask_df = dd.from_delayed(delayed_results)

    with ProgressBar():
        out = dask_df.compute()

    return out


def main_data_prep(lookback=252):
    # load and scrub the data-frame
    df = load_pkl_file()
    df = scrub_df()

    # compute residual returns and concat all dates
    rr = residual_returns(df, lookback_window=lookback)

    print("Saving Residual Returns to Parqeut...")
    rr.to_parquet("residual_returns.parquet")


def run_example_algo():
    df = load_pkl_file()
    df = scrub_df(df)

    corrs = generate_sample_corr_mat(df)
    for i in corrs.keys():
        corr1 = corrs[i]
        pos, neg = make_pos_neg(corr1)
        c = Cluster((pos, neg))
        labels = c.SPONGE(k=20)
        generate_sorted_corr_matx(corr1, labels)

        
if __name__ == '__main__':
    N_LOOKBACK = 252
    main_data_prep(lookback=N_LOOKBACK)






   




   

        






   
    

