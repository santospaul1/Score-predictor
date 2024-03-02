import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests  # For making API requests

# Function to fetch data from football-data.org
def fetch_data_from_api(api_token, competition_id, season):
    # API endpoint for Premier League matches
    api_url = f"https://api.football-data.org/v2/competitions/{competition_id}/matches?season={season}"
    headers = {"X-Auth-Token": api_token}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['matches']
    else:
        print(f"Error: {response.status_code} while retrieving data for season {season}")
        return []

# Function to preprocess API data
def preprocess_api_data(api_data):
    api_df = pd.DataFrame(api_data)
    api_df = api_df[['homeTeam', 'awayTeam', 'score']]

    # Process API data
    api_df['HomeTeam'] = api_df['homeTeam'].apply(lambda x: x['name'])
    api_df['AwayTeam'] = api_df['awayTeam'].apply(lambda x: x['name'])
    
    # Extract full-time home goals (FTHG) and full-time away goals (FTAG)
    api_df['FTHG'] = api_df['score'].apply(lambda x: x['fullTime']['homeTeam'])
    api_df['FTAG'] = api_df['score'].apply(lambda x: x['fullTime']['awayTeam'])
    
    # Extract match result (FTR)
    api_df['FTR'] = api_df.apply(lambda row: f"{row['FTHG']}-{row['FTAG']}", axis=1)
    
    return api_df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

# Competition ID for Premier League
competition_id = "PL"

# Retrieve data for all seasons from 2019 to 2024
all_data = []
for season in range(2019, 2024):
    # Replace with your API token
    api_token = "8c27542ed83f4b0492db46a921ba8dd1"
    season_data = fetch_data_from_api(api_token, competition_id, season)
    all_data.extend(season_data)

# Check if any data was retrieved
if not all_data:
    print("No data retrieved from the API. Please check your API token and internet connection.")
    exit()

# Preprocess and combine data from all seasons
api_df = preprocess_api_data(all_data)

# Load Premier League dataset
df = api_df.dropna(subset=['FTHG', 'FTAG'])  # Drop rows with missing goals

# Selecting relevant columns
X = df[['HomeTeam', 'AwayTeam']]
y_classification = df['FTR']  # Match result
y_regression = df[['FTHG', 'FTAG']]  # Goals

# Label encoding for 'Home Team' and 'Away Team'
label_encoder_X = LabelEncoder()
X.loc[:, 'HomeTeam'] = label_encoder_X.fit_transform(X['HomeTeam'])
X.loc[:, 'AwayTeam'] = label_encoder_X.transform(X['AwayTeam'])

# Encode match result 'FTR' using label encoder
label_encoder_y_classification = LabelEncoder()
y_classification_encoded = label_encoder_y_classification.fit_transform(y_classification)

# Split the dataset into training and testing sets
X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification_encoded, test_size=0.2, random_state=42)

# Create a Random Forest classifier for match result prediction
classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_model.fit(X_train, y_train_classification)

# Function to predict match result
def predict_match_result(home_team, away_team):
    try:
        # Encode team names
        home_team_encoded = label_encoder_X.transform([home_team])[0]
        away_team_encoded = label_encoder_X.transform([away_team])[0]
        
        # Create input data for prediction
        match_data = [[home_team_encoded, away_team_encoded]]
        
        # Make prediction
        prediction = classifier_model.predict(match_data)
        predicted_result = label_encoder_y_classification.inverse_transform(prediction)[0]
        
        return predicted_result
    except ValueError as e:
        print(f"Error: {e}")
        return "Unknown"

# Input example: Man United vs Leeds
home_team_input = "Fulham FC"
away_team_input = "Brighton & Hove Albion FC"

# Predict the match result
predicted_result = predict_match_result(home_team_input, away_team_input)
print(f"Predicted result for {home_team_input} vs {away_team_input}: {predicted_result}")

# Split the dataset into training and testing sets for regression
X_train, X_test, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Create separate Random Forest regressors for each goal feature
regressor_models = {}
for column in y_regression.columns:
    regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor_model.fit(X_train, y_train_regression[column])
    regressor_models[column] = regressor_model

# Function to predict goals
def predict_goals(home_team, away_team):
    try:
        # Encode team names
        home_team_encoded = label_encoder_X.transform([home_team])[0]
        away_team_encoded = label_encoder_X.transform([away_team])[0]
        
        # Create input data for prediction
        match_data = [[home_team_encoded, away_team_encoded]]
        
        # Make predictions for each goal feature
        predictions = {}
        for column, model in regressor_models.items():
            prediction = model.predict(match_data)
            rounded_prediction = round(prediction[0])  # Round to the nearest integer
            predictions[column] = rounded_prediction
        
        return predictions
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Function to determine the overall winner
def determine_overall_winner(home_team, away_team):
    home_team_goals = 0
    away_team_goals = 0
    
    # Simulate multiple matches and count goals scored
    num_simulations = 100
    for _ in range(num_simulations):
        goals = predict_goals(home_team, away_team)
        if goals is not None:
            home_team_goals += goals['FTHG']
            away_team_goals += goals['FTAG']
    
    # Compare the total goals scored
    if home_team_goals > away_team_goals:
        return home_team
    elif home_team_goals < away_team_goals:
        return away_team
    else:
        return "Draw"

# Predict goals
predicted_goals = predict_goals(home_team_input, away_team_input)

# Determine the overall winner
overall_winner = determine_overall_winner(home_team_input, away_team_input)
print(f"Overall winner between {home_team_input} and {away_team_input}: {overall_winner}")
print(f"Predicted goals for {home_team_input} vs {away_team_input}: {predicted_goals}")
