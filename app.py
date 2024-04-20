import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sqlite3 as sql

# connect to database
conn = sql.connect("ml_results.db")
st.title("Cumulative Returns of Cluster Selection Strategy")

# cumulative returns 
df = pd.read_sql('Select * from strategy_returns', conn)
metric_df = pd.read_sql("Select * from metrics_df", conn)

# set both indices
df.set_index('date', inplace=True)
metric_df.set_index('index', inplace=True)


# User input to select strategies
selected_strategies = st.multiselect(
    'Select investment strategies to plot:',
    options=df.columns.to_list(),
    default=df.columns[0]
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