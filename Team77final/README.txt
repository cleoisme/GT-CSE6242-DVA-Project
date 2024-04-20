DESCRIPTION - This project is a combination of using cluster via SPONG over a large universe of US Equity returns and then forecasting mean cluster returns over the next period using time-series momentum features and machine learning.
Finally we produce performance results using quartiles based on our forecasts to pick which clusters to over/under weight. For visualization we provide a simpel streamlit app of the results and also an interactive cluster graph using d3.js. 


INSTALLATION - How to install and setup the code:
1: Clone the repository here, https://github.com/cleoisme/GT-CSE6242-DVA-Project
2: Download the project data from the sharepoint. You must be in the domain of gatech.edu. https://gtvault-my.sharepoint.com/:u:/g/personal/jmaniery3_gatech_edu/EeT-rOHzKhpIrE3KeVHCkUsBtjk7jxdy5QvXPsk9aSFJWQ?e=LCYTAZ
   a: Extract the zip folder in the root directory of the cloned repository
3: We need pip installed within a conda environment to run the requirements.txt
   a: Create conda environment via "conda create -n python=3.11.5 yourenv pip" (this will make sure pipe is installed)
   b: Active conda environment, navigate to the repository and run "pip install -r requirements.txt"
   c: For reference use https://datumorphism.leima.is/til/programming/python/python-anaconda-install-requirements/


EXECUTION - after installation open an anaconda prompt and do the following steps:
1: Acitvate the conda environment created above
2: Navigate to the project root directory 
3: type "streamlit run app.py"

Comments: This will run the streamlit app in browser where you will have access to the cluster forecasting returns in an interactive line plot,
a table and finally the cross validation results of our five year expanding window custom method. This is similar to sklearn TimeSeriesSplit (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
with the difference being we define the splits by date corresponding to five year increments. 