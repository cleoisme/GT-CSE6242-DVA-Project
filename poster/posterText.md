# Title

SPONGE and Beyond: Signed Weighted Graph Stock Clustering for Dynamic Market Forecasting

# Distribution

## 10% Motivation/Introduction:

- 5% What is the problem (no jargon)?

  As a popular trading strategy, statistical arbitrage has evolved over decades, which typically rely on correlation-based approaches and may overlook negative relationships or fail to distinguish between positive and negative interactions. The SPONGE algorithem, a graph clustering implementation for constructing portfolios, marks a major step-forward to the traditional statistical arbitrage by addressing those limitations. In addition, the algorithm implementation is difficult for many people to follow, and thus creates a high technical barrier to keep potential investors out of the market.

- 5% Why is it important and why should we care?

  SPONGE's superior returns to traditional statistical arbitrage are certainly a good thing for investors. And our project is important in lowering the investment threshold and encouraging potential investors to enter the market. More investors mean more money in the flow, which will be of great benefit to the of the financial development.

## 20% Your approaches (algorithm and interactive visualization):

- 5% What are they?

  SPONGE and D3.js

- 5% How do they work?

  - SPONGE:

    SPONGE utilizes the spectral properties of the Laplacian matrix associated with the signed network to identify clusters. By solving this generalized eigenvalue problem, SPONGE effectively partitions the network into clusters while considering both positive and negative interactions between nodes. This approach enables SPONGE to handle recent stock data using signed networks for our project, where edges represent both positive and negative relationships, and effectively uncover meaningful clusters within such networks (groups of similar companies).

    [To Insert: a figure explaining how the algo works]

  - D3.js:

    D3 will be used to visualize the correlation matrix of stock residual returns as a network, which will introduce the intuition, interactivity and the user-friendness to our project.

    [To Insert: a figure explaining how D3.js represent the data from SPONGE and how users can interact with the visualization]

- 5% Why do you think they can effectively solve your problem (i.e., what is the intuition behind your approaches)?

  SPONGE presents a novel approach to clustering signed networks, offering advantages over traditional statistical arbitrage strategies, by formulating the clustering problem as a generalized eigenvalue problem and leveraging the spectral properties of the Laplacian matrix associated with the network. SPONGE effectively identifies clusters while considering both positive and negative interactions between nodes, enhancing its applicability to diverse networks and promotes robustness to noise, providing interpretable clustering results.

- 5% What is new in your approaches?

  In addition to replicating the SPONGE algorithm we plan to visualize the correlation matrix of stock residual returns as a network using d3.js. Additionally, we aim to build a time-series forecasting model to choose the best cluster to invest in over the next period for long-term investor.

## 10% Data:

- 5% How did you get it? (Download? Scrape?)

  The data is provided by a industry practioner, which can be collected publically as well.

- 5% What are its characteristics (e.g., size on disk, # of records, temporal or not, etc.)
  - Size: 1.73GB
  - Record Number: (37104728, 10)
  - Permanant data and updated till March 2024

## [Need Help] 25% Experiments and results:

- 5% How did you evaluate your approaches?

  1. Hypothesis: Long/short mean cluster forecasting strategy can outperform the mean return of our investment universe while taking the same amount of risk.
  2. Design matrix X are time-series momentum features for each cluster on each day and targets are their associated one-day forward risk adjusted return. This is a regression problem.
  3. Expanding window walk-forward cross validation at 5 year increments using a 90/10 split for training and validation (tuning hyper-parameters), testing set five years forward. For parameter search we plan to use randomized grid-search in sklearn.
  4. Modeling : Use linear models as a baseline and compare with more complex machine model (e.g Ridge vs Random Forest)
  5. We will have multiple evaluation criteria, for tuning in the walk-forward step we will use **mean-squared error (MSE)**. For investment performance we will evaluate metrics like **total return** and **Sharpe ratio**. We expect to see a linear relationship between the accuracy of the forecasting models and investment performance

- 10% What are the results?
  [TODO]
- 10% How do your methods compare to other methods?
  [Traditional arbitrage methods? QQQ?]

## 10% Presentation delivery:

- 5% Finished on time?
- 5% Spoke clearly and at a good pace?

## 25% Poster Design:

- 5% Layout/organization (Clear headings? Easy to follow?)
- 5% Use of text (Succinct or verbose?)
- 5% Use of graphics (Are they relevant? Do they help you better understand the project's approaches and ideas?)
- 5% Legibility (Is the text and figures too small?)
- 5% Grammar and spelling
