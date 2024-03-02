<h1>Football Match Prediction Using Decision Trees</h1>

This repository contains Python code that utilizes Decision Trees to predict football match outcomes and goals. It leverages data from the Football-Data.org API to build the model.

<h2>Prerequisites:</h2>
<ul>
<li>Python 3.x (https://www.python.org/downloads/)</li>
<br >
<li>pandas (https://pandas.pydata.org/docs/getting_started/install.html)</li>
<br >
<li>scikit-learn (https://scikit-learn.org/)</li>
<br >
<li>requests (https://requests.readthedocs.io/)</li>
  </ul>
<h2>Instructions:</h2>

Install required libraries:

```Bash
pip install pandas scikit-learn requests
```


<h4>Replace API token:</h4>

Create an account on https://www.football-data.org/documentation/api to obtain a free API token.<br >
Replace YOUR_API_TOKEN in the code with your actual token to retrieve data.<br >
<strong>Run the script:</strong>

```Bash
python correct_score_predictor.py
```

<h2>Description:</h2>
<ol>
<li>The code retrieves match data from the Football-Data.org API for Premier League seasons or any league as per your api request between 2019 and 2024 (inclusive).</p>
<li>It preprocesses the data, including handling missing values and applying label encoding for categorical features.</li>
<li>It builds separate decision tree models:</li>
<li>One for predicting match results (e.g., "Home Win", "Draw", "Away Win")</li>
<li>One for predicting the number of goals scored by each team (home and away)</li>
<li>It simulates multiple matches based on the predicted goals and determines the overall winner (based on total goals scored).</li>
  </ol>
<strong>Note:</strong> This is a simplified example for demonstration purposes. Machine learning models have limitations and uncertainties. Use them cautiously and interpret predictions as possible outcomes, not absolute guarantees.
<h3>Further Enhancements:</h3>

<p>Explore more sophisticated models (e.g., Random Forests, Support Vector Machines) for potentially better prediction accuracy.</p>
<p>Implement functionalities to allow users to input specific teams and predict outcomes for upcoming matches.</p>
<p>Integrate the model into a user interface for interactive exploration and prediction.</p>
<h2>Disclaimer:</h2>

The provided API token is for demonstration purposes only. Please obtain your own token from Football-Data.org and comply with their terms of use.

