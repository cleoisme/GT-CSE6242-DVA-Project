## Description
This project is a combination of using clustering via SPONGE over a large universe of US Equity returns and then forecasting mean cluster returns over the next period using time-series momentum features and machine learning.
Finally, we produce performance results using quartiles based on our forecasts to pick which clusters to overweight or underweight. For visualization we provide a simple streamlit app of the results and also an interactive cluster graph using d3.js. 

## Installation
To install and set up the codebase, follow these steps:

1. Clone the GitHub repository at https://github.com/cleoisme/GT-CSE6242-DVA-Project.
2. Download the project data from SharePoint, accessible only within the gatech.edu domain at https://gtvault-my.sharepoint.com/:u:/g/personal/jmaniery3_gatech_edu/EeT-rOHzKhpIrE3KeVHCkUsBtjk7jxdy5QvXPsk9aSFJWQ?e=LCYTAZ. Ensure to:
   a. Extract the ZIP folder into the root directory of the cloned repository.
3. Use pip within a conda environment to process the `requirements.txt` by:
   a. Creating a conda environment with `conda create -n your_env_name python=3.11.5 pip`.
   b. Activating the conda environment and navigating to the repository to run `pip install -r requirements.txt`.
   c. Reference the installation guide at https://datumorphism.leima.is/.

## Execution
Post-installation, to execute the application:

1. Activate the previously created conda environment.
2. Navigate to the project's root directory.
3. Execute the Streamlit app by typing `streamlit run app.py` into the Anaconda prompt.

## Additional Information:
Launching the Streamlit app provides access to an interactive line plot representing cluster forecasting returns, a corresponding table, and cross-validation results employing a custom five-year expanding window method akin to sklearn's TimeSeriesSplit, differentiated by date-defined splits that align with five-year increments.
