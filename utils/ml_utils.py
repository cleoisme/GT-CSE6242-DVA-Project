import empyrical as ep
import numpy as np
import pandas as pd



def build_features(data, groupby_key="cluster"):
    """
    builds the momentum features mentioned in the paper, 
    """
    # make copy
    data = data.copy()

    # ewm realized volatility a rough forecast of t+1
    data['rVol'] = data.groupby(by='cluster')[['ret']].pct_change().ewm(span=60).std()

    # trailing returns
    data['1d_ret'] = data.groupby(by=groupby_key)['ret'].pct_change(1)
    data['1wk_ret'] = data.groupby(by=groupby_key)['ret'].pct_change(5)
    data['1m_ret'] = data.groupby(by=groupby_key)['ret'].pct_change(20)
    data['1Q_ret'] = data.groupby(by=groupby_key)['ret'].pct_change(60)
    data['6M_ret'] = data.groupby(by=groupby_key)['ret'].pct_change(124)
    data['12M_ret'] = data.groupby(by=groupby_key)['ret'].pct_change(252)

    # build risk adjusted features
    data['feature_1d_ra'] = data['1d_ret']/data['rVol']
    data['feature_1wk_ra'] = data['1wk_ret']/(data['rVol'] * np.sqrt(5))
    data['feature_1m_ra'] = data['1m_ret']/(data['rVol'] * np.sqrt(20))
    data['feature_1Q_ra'] = data['1Q_ret']/(data['rVol'] * np.sqrt(60))
    data['feature_6M_ra'] = data['6M_ret']/(data['rVol'] * np.sqrt(124))
    data['feature_12M_ra'] = data['12M_ret']/(data['rVol'] * np.sqrt(252))

    # build moving-average convergence divergence features
    data['feature_MACD_short'] = (data.groupby(by=groupby_key)['ret'].ewm(span=8).mean() - data.groupby(by=groupby_key)['ret'].ewm(span=24).mean()).droplevel(0)/data.groupby(by=groupby_key)['ret'].ewm(span=63).std().droplevel(0)
    data['feature_MACD_medium'] = (data.groupby(by=groupby_key)['ret'].ewm(span=16).mean() - data.groupby(by=groupby_key)['ret'].ewm(span=48).mean()).droplevel(0)/data.groupby(by=groupby_key)['ret'].ewm(span=63).std().droplevel(0)
    data['feature_MACD_long'] = (data.groupby(by=groupby_key)['ret'].ewm(span=32).mean() - data.groupby(by=groupby_key)['ret'].ewm(span=96).mean()).droplevel(0)/data.groupby(by=groupby_key)['ret'].ewm(span=63).std().droplevel(0)

    # now for new features
    data['feature_skew6m'] = data.groupby(by=groupby_key)['ret'].pct_change(1).rolling(124).skew()
    data['feature_skew12m'] = data.groupby(by=groupby_key)['ret'].pct_change(1).rolling(252).skew()
    data['feature_kurt6m'] = data.groupby(by=groupby_key)['ret'].pct_change(1).rolling(124).kurt()
    data['feature_kurt12m'] = data.groupby(by=groupby_key)['ret'].pct_change(1).rolling(252).kurt()

    # Create lagged features
    _features = [f for f in data.columns if f.startswith('feature')]
    for lag in [1, 2, 3, 4, 5]:
        for feat in _features:
            data[f'lag{lag}_{feat}'] = data.groupby(by=groupby_key)[feat].shift(lag)

    # also build the target - target is +1D risk adjusted return
    data['fwd_ret1d'] = data.groupby(by=groupby_key)['1d_ret'].shift(-1)
    data['target'] = data['fwd_ret1d']/data['rVol']
    data['targetBin'] = np.sign(data['target'])

    return data



def get_returns_breakout(strats: pd.DataFrame):
    """
    tuple of (strat name, series of returns)
    """

    ret_breakout=pd.DataFrame(columns=['Annual_Return', 'Annual_Volatility',
                                       'DD', 'MDD', 'Sharpe', 'Sortino',
                                       'Calmar', 'ppct_postive_rets'])
    for strat in strats.columns:
        _strat = strats[strat]
        ret_breakout.loc[strat, 'Annual_Return'] = ep.annual_return(_strat)
        ret_breakout.loc[strat, 'Annual_Volatility'] = ep.annual_volatility(_strat)
        ret_breakout.loc[strat, 'DD'] = ep.downside_risk(_strat)
        ret_breakout.loc[strat, 'MDD'] = ep.max_drawdown(_strat)
        ret_breakout.loc[strat, 'Sharpe'] = ep.sharpe_ratio(_strat)
        ret_breakout.loc[strat, 'Sortino'] = ep.sortino_ratio(_strat)
        ret_breakout.loc[strat, 'Calmar'] = ep.calmar_ratio(_strat)
        ret_breakout.loc[strat, 'ppct_postive_rets'] = _strat[_strat>0].shape[0]/_strat.shape[0]

    return ret_breakout


def cv_date_splitter(dates: list, split_length: int=252 * 5) -> list:
    """
    returns time points for expanding window cross-valiation (start, end, test)
    """
    out = []
    start, end = None, None
    num_splits = len(dates)//split_length
    print(num_splits)

    for k, split in enumerate(range(num_splits)):
        if k==0:
            start = dates[0]
            end = dates[split_length]
            out.append((start, end, dates[split_length *(k+2)]))

        elif k>0 and k<num_splits-1:
            start = dates[0]
            end = dates[split_length*(k+1) + 1]
            out.append((start, end, dates[split_length *(k+2)]))

        elif k==num_splits-1:
            start = dates[0]
            end_first = dates[split_length*(k+1) + 1]
            out.append((start, end_first, dates[-1]))
    return out


def get_cv_splits(feats: pd.DataFrame, split_length: int=252*5):
    """
    yields train, test splits that can be used in the evaulation loop
    """
    # get the splits
    splits = cv_date_splitter(feats.index.get_level_values('date').unique(), split_length=split_length)

    for split in splits:
        train = feats.loc[(feats.index.get_level_values('date')>=split[0]) & (feats.index.get_level_values('date')<=split[1])]
        test = feats.loc[(feats.index.get_level_values('date')>split[1]) & (feats.index.get_level_values('date')<=split[2])]
        yield train, test


def train_val_split(X_train: pd.DataFrame, y_train: pd.DataFrame, train_pct:float=.90):
    # train split
    train_split = round(X_train.shape[0] * train_pct)

    # Xtrain and ytrain
    X_train2 = X_train.head(train_split)
    y_train2 = y_train.head(train_split)
    last_train_date = X_train2.index.get_level_values('date')[-1]

    # X and y validation
    X_val = X_train.loc[X_train.index.get_level_values('date')>last_train_date]
    y_val = y_train.loc[y_train.index.get_level_values('date')>last_train_date]

    return X_train2, X_val, y_train2, y_val