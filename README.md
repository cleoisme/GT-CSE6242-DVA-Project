# GT-CSE6242-DVA-Project

## 2024-02-18 Init Meeting
### Work Distribution:
|                            | % Grades |   Due Date  |        | 
| -------------------------- | -------- | ----------- |--------| 
|          Proposal          |   7.5%   | Fri, Mar 1  |        | 
|    Proposal presentation   |    5%    | Fri, Mar 1  |Jonathan| 
|       Progress report      |    5%    | Fri, Mar 29 |        | 
|  Final poster presentation |   7.5%   | Fri, Apr 19 |  Cleo  | 
|        Final report        |   25%    | Fri, Apr 19 |        | 

### [Project Description](https://docs.google.com/document/u/0/d/e/2PACX-1vSlYrMw402tL3F95ay-AaptTdF80UOER-gne_O0kqbuuk6WXrlsjwaYjjS0Jyl95dXYyDLjh9DR1mln/pub?pli=1)
9 Heilmeier questions need to be answered by the proposal:

1. What are you trying to do? Articulate your objectives using absolutely no jargon.
2. How is it done today; what are the limits of current practice?
3. What's new in your approach? Why will it be successful?
4. Who cares?
5. If you're successful, what difference and impact will it make, and how do you measure them (e.g., via user studies, experiments, ground truth data, etc.)?
6. What are the risks and payoffs?
7. How much will it cost?
8. How long will it take?
9. What are the midterm and final "exams" to check for success? How will progress be measured?

### Ideas:
1. Core CPI forecaster:
<br /> Instead of predicting the trend of cpi, I am thinking about something more like a model to assess the current macro-economic, showing the impact of different factors in the form of bubbles and giving homomorphic outcomes.
<br /> There are several statistics we can leverage:
    - Core CPI
    - non-farm payrolls
    - GDP
    - Interest rate
    - layoff numbers (?)
    - Bond price (?)
      
    Source: https://research.stlouisfed.org/econ/mccracken/fred-databases

2. The graph clustering (partially implementation) and Viz
<br /> Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4560455


3. Viz of a network of ___

## 2024-02-25 Literature Review and Idea Presentation

2.-JM
-The idea is to visualize the stock market as a network where the visualization application would probably be in d3 or Tableau  
      The paper uses SPONG clustering which is an algorithm for clustering a signed graph (we have positive and negative correlations)
<br /> Source2a: https://arxiv.org/abs/1904.08575
<br /> Source2b (Implementation): https://github.com/alan-turing-institute/SigNet/blob/master/README.md
<br /> Sources2c: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

      -Step 1: Gather individual stock data, we can use free open source yfinance to gather the data and store it in SQLite
      -Step 2: Compute residual returns - most stocks have exposure to the over-all market which we call "beta" and other movements which are not explained by the market 
               which we call alpha. This is just a linear regression y(stockA), x(market) and we extract a time-series of residuals for each stock on some rolling window.
      -Step 3: The clustering is based on a correlation matrix of the residual returns, this tells use what stocks move together based on their unique characteristics and not 
               the over-all market. 
      -Step 4: Transform the correlation matrix for clustering. We have a signed weighted graph, what are the issues here?
               A correlation matrix is a similarity matrix
      -Step 5: Apply the Specteral clustering/SPONG on the transformed matrix, tune the number of clusters
      -Step 6: Output and visualization: 1. d3 graph for a given time period of our data set (correlation matrix needs a look-back window to be calculated), 2. Line plot of 
               returns for each cluster over a given time period, probably want risk adjusted returns as well for comparability 3. Figure out a way to show the under/over valued
               stocked in each cluster as defined by under/out performance vs the mean of each cluster over some look-back period.

       1. What are you trying to do? Articulate your objectives using absolutely no jargon.
            Cluster similar stocks together with a correlation matrix to discover mis-priced securities
       2. How is it done today; what are the limits of current practice?
            Common approaches today would be something like using categorical assigned sectors, for example evaluate stocks in the Energy sector
       3. What's new in your approach? Why will it be successful?
            This approach with inspiration from a recent paper uses automated statistical learning to indentify complex clusters we might of not know existed.
            For example maybe the algorithm can automatically group together prodcuers and supplies. It could be successful under the hypothesis that most 
            investors are not using data mining/stastical learning techniques to parse relationships in the stock market. 
       4. Who cares?
            People who want to trade or invest, or even just have a better understanding of the stock market and it's interactions. 
       5. If you're successful, what difference and impact will it make, and how do you measure them (e.g., via user studies, experiments, ground truth data, etc.)?
            1. If successful we hope users will gain a better understand of diversification and risks they are taking. For example instead of just buying one stock an investor could buy all the stocks in a given cluster to avoid company specific risk.
            2. In the spirit of the original paper this will help traders identify mispriced securities.  
       6. What are the risks and payoffs?
            Compute - if we use the entire stock market we will have thousands of stocks and computing the (N*N) correlation matrix will be slow
            Clustering Algorithm will need to be refit for each time-period (say the user selects a random continuous 2-year historical period)
       7. How much will it cost?
       8. How long will it take?
       9. What are the midterm and final "exams" to check for success? How will progress be measured?



 DL - impact of economic and political events on FOREX markets (and investment recommendations)
 
- Gather FOREX data for # major currencies
- Gather/score political and economical events in and between countries 
- Show influence of current and futures of the FOREX pair
- Recommend investment options

Data sources can be Bloomberg FOREX (need to confirm)
Dashboard can be in Tableau



References:

https://www.knowi.com/blog/analysis-the-impact-of-economic-and-political-events-on-global-markets/

Forex market forecasting using machine learning: Systematic Literature Review and meta-analysis
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00676-2

Exchange rates: the influence of political and economic events. а fundamental analysis approach 
https://www.researchgate.net/publication/329912469_Exchange_rates_The_influence_of_political_and_economic_events_A_fundamental_analysis_approach

