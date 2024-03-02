# GT-CSE6242-DVA-Project

## 2024-02-18 Init Meeting

### Work Distribution:

|                           | % Grades | Due Date    |          |
| ------------------------- | -------- | ----------- | -------- |
| [Proposal](https://www.overleaf.com/project/65debb9a9e68928c323848ef)                  | 7.5%     | Fri, Mar 1  |          |
| Proposal presentation     | 5%       | Fri, Mar 1  | Jonathan |
| Progress report           | 5%       | Fri, Mar 29 |          |
| Final poster presentation | 7.5%     | Fri, Apr 19 | Cleo     |
| Final report              | 25%      | Fri, Apr 19 |          |

### [Project Description](https://docs.google.com/document/u/0/d/e/2PACX-1vSlYrMw402tL3F95ay-AaptTdF80UOER-gne_O0kqbuuk6WXrlsjwaYjjS0Jyl95dXYyDLjh9DR1mln/pub?pli=1)

Answer the 9 Heilmeier questions in the proposal:

1. What are you trying to do? Articulate your objectives using absolutely no jargon.
2. How is it done today; what are the limits of current practice?
3. What's new in your approach? Why will it be successful?
4. Who cares?
5. If you're successful, what difference and impact will it make, and how do you measure them (e.g., via user studies, experiments, ground truth data, etc.)?
6. What are the risks and payoffs?
7. How much will it cost?
8. How long will it take?
9. What are the midterm and final "exams" to check for success? How will progress be measured?

### 2024-02-25 Ideas:

1. CZ - Macroeconomic Evaluator

    Goals:
    - Provide intuitive visualizations for people without an economics background.
    - Visualize the interrelatedness of key economic indicators and their impact on macroeconomics.
    - Make short-term predictions.

    Steps:
    1. Data Collection:
        - Identify key economic indicators.
        - Sources: [Federal Reserve Economic Data (FRED)](https://research.stlouisfed.org/econ/mccracken/fred-databases), [Bureau of Economic Analysis (BEA)](https://www.bea.gov/), [Bureau of Labor Statistics (BLS)](https://www.bls.gov/), [U.S. Census Bureau](https://www.census.gov/).
    2. Model Selection:
        - Consider machine learning algorithms like ARIMA, SARIMA, linear regression, Random Forest, Gradient Boosting, LSTM, or GRU.
    3. Train and Evaluate the Model:
        - Split data, train the model, and evaluate performance.
    4. Network Representation and Result Evaluation:
        - Create graph-based models to represent relationships between economic indicators.
    5. Interpretation:
        - Explain the significance of connections and patterns in the network.
    6. Concerns:
        - Evaluating relationships between indicators may be a substantial project on its own.
        - Accumulation of inaccuracies is unavoidable.
        - Balance between data visualization and algorithm complexity needs consideration.

2.  JM - Visualize the stock market as a network using SPONG clustering.
    - [Source 2a](https://arxiv.org/abs/1904.08575), [Source 2b (Implementation)](https://github.com/alan-turing-institute/SigNet/blob/master/README.md), [Sources 2c](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)

    Steps:
    1. Gather individual stock data using yfinance.
    2. Compute residual returns through linear regression.
    3. Use correlation matrix for clustering.
    4. Apply Spectral clustering/SPONG on the transformed matrix.
    5. Output and visualization: d3 graph, line plot of returns for each cluster, show under/overvalued stocks.

    Objectives:
    - Cluster similar stocks to discover mispriced securities.

    Risks and Payoffs:
    - Computing the correlation matrix for the entire stock market may be slow.
    - Clustering algorithm needs refitting for each time period.

3. DL - Impact of economic and political events on FOREX markets (and investment recommendations)

    Steps:
    - Gather FOREX data for # major currencies.
    - Gather/score political and economical events.
    - Show influence on current and future FOREX pairs.
    - Recommend investment options.

    Data sources: Bloomberg FOREX.
    Dashboard: Tableau.

    References:
    - [Analysis: The Impact of Economic and Political Events on Global Markets](https://www.knowi.com/blog/analysis-the-impact-of-economic-and-political-events-on-global-markets/)
    - [Forex Market Forecasting using Machine Learning: Systematic Literature Review and Meta-analysis](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00676-2)
    - [Exchange Rates: The Influence of Political and Economic Events](https://www.researchgate.net/publication/329912469_Exchange_rates_The_influence_of_political_and_economic_events_A_fundamental_analysis_approach)

4. AQ - Visualize EMS response times in a metro area.

    Objectives:
    - Provide people or local governments the ability to see response times by address.
    - Analyze network for the best possible response time.

    Datasets:
    - [Law Enforcement Dispatched Calls for Service](https://data.sfgov.org/Public-Safety/Law-Enforcement-Dispatched-Calls-for-Service-Real-/gnap-fj3t/about_data)
    - [Police Department Incident Reports 2018 to Present](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/about_data)
    - [San Francisco Neighborhoods GeoJSON](https://gist.githubusercontent.com/cdolek/d08cac2fa3f6338d84ea/raw/ebe3d2a4eda405775a860d251974e1f08cbe4f48/SanFrancisco.Neighborhoods.json)
    - [GeoJSON Dijkstra](https://github.com/royhobbstn/geojson-dijkstra) (Possible package for finding the best path)

    Papers:
    - [Optimizing EMS Response Times](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7718983/)
    - [Optimization of Emergency Medical Service Location](https://www.sciencedirect.com/science/article/pii/S2772442523001235)
    - [Effect of Ambulance Response Times on Patient Outcomes](https://jamanetwork.com/journals/jamasurgery/fullarticle/2643992)

    Main concerns:
    - Reliance on tools outside the class scope.
    - Analysis might be challenging in reality.
    - Proposed analysis may be redundant as GPS likely already has optimal routes calculated.

LITERATURE FOR OUR PRO
