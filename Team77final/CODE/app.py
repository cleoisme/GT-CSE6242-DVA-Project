import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sqlite3 as sql
from forecast import run_model
from utils.ml_utils import get_returns_breakout

def run_app():
    st.title("Cumulative Returns of Cluster Selection Strategy")

    def tweak_cv_res(df, model_name):
        df['split'] = [x for x in range(4)]
        df['model'] = model_name
        df.drop(columns=['split_0',
                        'split_1',
                        'split_2',
                        'split_3'],
                        axis=1,
                        inplace=True)
        return df


    MODELS = ['ElasticNet',
            'ElasticNetFS',
            'RandomForest',
            'RandomForestFS']


    # cumulative returns
    try:
        conn = sql.connect("ml_results.db") 
        df = pd.read_sql('Select * from strategy_returns', conn)
        metric_df = pd.read_sql("Select * from metrics_df", conn)
        preds_eNet = pd.read_pickle("preds_eNet_results.pkl")
        preds_eNetPCA = pd.read_pickle("preds_eNetPCA_results.pkl")
        preds_RF_FS = pd.read_pickle("preds_RF_FS_results.pkl")
        preds_RF = pd.read_pickle("preds_RF_results.pkl")
        dfs = [preds_eNet, preds_eNetPCA, preds_RF, preds_RF_FS]
        all_results = pd.concat([tweak_cv_res(x, model) for x, model in zip(dfs, MODELS)])
        mean_scores = all_results.groupby(by='model')['scores'].mean()
        std_scores = all_results.groupby(by='model')['scores'].std()
        agg_results = mean_scores.to_frame('meanCV_score').join(std_scores.to_frame('stdCV_score'))
        agg_results['coef_of_variation'] = agg_results['meanCV_score']/agg_results['stdCV_score']
    except Exception as e:
        print('Model data not found, running model....')
        run_model()

    # set both indices
    df.set_index('date', inplace=True)
    metric_df.set_index('index', inplace=True)

    # create a variable for the market
    df['market'] = (1+df.pct_change().mean(axis=1)).cumprod()
    mkt_stats = get_returns_breakout(df[['market']].pct_change().dropna()).reset_index()
    mkt_stats.set_index('index', inplace=True)
    metric_df = pd.concat([metric_df, mkt_stats])


    # User input to select strategies
    selected_strategies = st.multiselect(
        'Select investment strategies to plot:',
        options=df.columns.to_list(),
        default=['preds_RF_Q_3.0', 'preds_eNet_Q_3.0', 'market']
    )

    # Filter the DataFrame based on the selected strategies
    if selected_strategies:
        df = df[selected_strategies]
        # Melt the cumulative returns for the selected strategies into long format for Altair
        data_long = df.reset_index().melt('date', var_name='strategy', value_name='cumulative_return')

        # Create a multi-line chart
        chart = alt.Chart(data_long).mark_line().encode(
            x='date:T',
            y=alt.Y('cumulative_return:Q', axis=alt.Axis(title='Cumulative Returns')),
            color='strategy:N',
            tooltip=['date:T', 'cumulative_return:Q', 'strategy:N']
        ).interactive().properties(width=700, height=400)

        # Display the chart in Streamlit
        st.altair_chart(chart)
    else:
        st.write("Please select at least one strategy to plot.")


    # Display the performance statistics table under the line chart
    st.write('Performance Statistics for Cluster Strategies')
    st.dataframe(metric_df.style.format("{:.2f}"))

    #################################################################
    # Display the cross-validation results as well?
    st.title("Model MSE by Cross-validation Split")

    selected_models = st.multiselect('Select Models:', options=all_results['model'].unique(),
                                    default="ElasticNet")

    if selected_models:
        # Filter the DataFrame based on selected models
        filtered_df = all_results[all_results['model'].isin(selected_models)]

        # Create a grouped bar chart
        chart = alt.Chart(filtered_df).mark_bar().encode(
            # The trick is to use 'split' as the main category and 'model' as a subcategory
            x=alt.X('split:N', axis=alt.Axis(title='Split')),
            y=alt.Y('scores:Q', axis=alt.Axis(title='Scores')),
            color='model:N',
            column=alt.Column('model:N', header=alt.Header(title='Model')),  # This separates the bars by model
            tooltip=['model:N', 'scores:Q', 'split:N']
        ).properties(
            width=alt.Step(20)  # Controls the width of each bar
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Please select at least one model to display the chart.")


    st.write('Aggregate CV Scores')
    st.dataframe(agg_results.style.format("{:.4f}"))

if __name__ == '__main__':
    run_app()