import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from utils.ml_utils import *

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from data_prep import load_features_data
from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA
import sqlite3 as sql


def prep_data():
    feats = load_features_data()
    feats['target'] = feats['target'].clip(-10., 10.)
    features = [f for f in feats.columns if f.startswith("feature")]
    target = ["target"]
    lag_feats = [f for f in feats.columns if f.startswith('lag')]
    features = [f for f in feats.columns if f.startswith("feature")]
    target = ["target"]
    lag_feats = [f for f in feats.columns if f.startswith('lag')]

    # make a new feature vector for features + lags
    all_feats = features + lag_feats + target
    feats.dropna(subset=all_feats, inplace=True)
    feats = feats[all_feats]
    X = feats[all_feats].copy()
    return X, features


def prep_models():
    N_ITER=50

    # Random Forest Regressor
    baseRF = RandomForestRegressor(max_depth=5, 
                                n_estimators=1000,
                                max_features=int(1),
                                random_state=49, 
                                n_jobs=-1)

    # simple-grid
    grid = {'rf__n_estimators': np.arange(100, 2000, 100),
            'rf__max_depth': [3, 6, 9, 12],
            'rf__max_features': [int(1), 'sqrt'],
            'rf__min_weight_fraction_leaf': np.arange(0.0, 0.05, 0.005)}
    paramsRF = ParameterSampler(n_iter=N_ITER, param_distributions=grid)

    rfPipe = Pipeline(steps=[('scaler', RobustScaler()), ('rf',baseRF)])
    rfPipeFS = Pipeline(steps=[('scaler', RobustScaler()),
                            ('selector', SelectFromModel(baseRF, threshold='mean')),
                            ('rf',baseRF)])

    # Elastic Net Regressor
    net = ElasticNet(max_iter=1000)
    eNetPipe = Pipeline(steps=[('scaler', RobustScaler()), ('net', net)])
    eNetPipePCA = Pipeline(steps=[('scaler', RobustScaler()),
                                ('pca', PCA(n_components=.95)),
                                ('net', net)])

    eNet_grid = {'net__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                'net__l1_ratio': np.arange(0.10, .90, 0.01)}
    paramsNet = ParameterSampler(n_iter=N_ITER, param_distributions=eNet_grid)

    # append the models
    models = [(eNetPipe, paramsNet, 'preds_eNet'),
              (eNetPipePCA, paramsNet, 'preds_eNetPCA'),
              (rfPipe, paramsRF, 'preds_RF'),
              (rfPipeFS, paramsRF, 'preds_RF_FS')]
    return models


def train_single_model(model, sampler, model_name, X, features):
        predictions = []
        scores = []
        res = []
        for c, (train, test) in tqdm(enumerate(get_cv_splits(X, split_length=252*5))):
            # break out X and y train, test
            X_train, y_train = train[features], train['target'] 
            X_test, y_test = test[features], test['target']

            # hyper-param loop
            X_train2, X_val, y_train2, y_val = train_val_split(X_train, y_train)
            print(X_train2.shape, X_val.shape)

            # inner loop for parameter tuning
            gscv_scores = {'scores': [], 'grid':[]}
            for k, p in enumerate(sampler):
                model.set_params(**p)
                try:
                    model.n_jobs=-1
                except Exception as e:
                    pass
                model.fit(X_train2, y_train2.values.reshape(y_train2.shape[0], ))
                _pred = model.predict(X_val)
                _score = mean_squared_error(y_val, _pred)
                gscv_scores['scores'].append(_score)
                gscv_scores['grid'].append(p)
                print(f'Iter: {k}: Score: {_score}')

            # now fit the best model
            best_model = pd.DataFrame(gscv_scores).sort_values(by='scores').head(1)['grid'].values[0]
            _res = pd.DataFrame(gscv_scores).sort_values(by='scores').head(1) # this should be the best score in the GSCV
            _res[f'split_{c}'] = c
            res.append(_res)
            print(pd.DataFrame(gscv_scores).sort_values(by='scores').head(1))
            print(f'Best Model: {best_model}')
            best_model = model.set_params(**best_model)
            best_model.n_jobs=-1
            best_model.fit(X_train, y_train.values.reshape(y_train.shape[0], ))
            preds = best_model.predict(X_test)

            # append the predictions
            predictions.append(pd.Series(index=y_test.index, data=preds))
        
            # score
            scores.append(mean_squared_error(y_test, preds))

        # predictions
        predictions = pd.concat(predictions).to_frame(model_name)
        return predictions, res

def run_model():
    # bring in the data
    X, features = prep_data()
    models = prep_models()
    
    predictions = []
    for model in models:
        out = train_single_model(model[0], model[1], model[2], X, features)
        out[1].to_pickle(f'{model[2]}_results.pkl')
        predictions.append(out[0])

    # concat the predictions by columns
    predictions = pd.concat(predictions, axis=1)

    # Bring in the data
    data = load_features_data()

    # join in predictions
    predictions = predictions.merge(data, how="left", left_index=True, right_index=True)
    predictions.sort_index(inplace=True)

    # get the prediction columns
    pred_cols = [f for f in predictions.columns if f.startswith('preds_')]

    def getRetsForStrat(predictions, pred_cols):
        ret_list = []
        for pred_col in pred_cols:
            predictions[f'{pred_col}_signalQ'] = predictions.groupby(by='date')[pred_col].apply(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')).droplevel(0)
            predictions[f'{pred_col}_signalQ'] = predictions.groupby(by='cluster')[f'{pred_col}_signalQ'].shift(1)
            rets = predictions.groupby(by=['date', f'{pred_col}_signalQ'])['1d_ret'].mean().unstack()
            rets.rename(columns={col: f'{pred_col}_Q_{col}' for col in rets.columns}, inplace=True)
            ret_list.append(rets)
        return ret_list

    ret_breakouts = []
    # get all the returns
    all_rets = getRetsForStrat(predictions, pred_cols)
    for one_ret in all_rets:
        print(get_returns_breakout(one_ret))
        ret_breakouts.append(get_returns_breakout(one_ret))
        print('#################################')

    # stack the returns
    stacked_strat_rets = pd.concat([r.stack().to_frame('ret') for r in all_rets]) # this can hopefully be the data used for streamlit
    stacked_strat_rets_cumprod = (1+stacked_strat_rets.unstack()).cumprod()

    # save to sqlite3 DB
    conn = sql.connect("ml_results.db")

    stacked_strat_rets_cumprod.columns = stacked_strat_rets_cumprod.columns.droplevel(0)
    stacked_strat_rets_cumprod.to_sql('strategy_returns', conn, if_exists='replace', index=True)
    pd.concat(ret_breakouts).to_sql("metrics_df", conn, if_exists='replace', index=True)


if __name__ == "__main__":
    run_model()

    