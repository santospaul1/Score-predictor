import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '01-22.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Selecting relevant columns
X = df[['HomeTeam', 'AwayTeam']]
y_classification = df['FTR']  # Match result
y_regression = df[['FTHG', 'FTAG', 'HTHG', 'HTAG']]  # Goals

# Label encoding for 'Home Team' and 'Away Team'
label_encoder_X = LabelEncoder()
X.loc[:, 'HomeTeam'] = label_encoder_X.fit_transform(X['HomeTeam'])
X.loc[:, 'AwayTeam'] = label_encoder_X.transform(X['AwayTeam'])

# Encode match result 'FTR' using label encoder
label_encoder_y_classification = LabelEncoder()
y_classification_encoded = label_encoder_y_classification.fit_transform(y_classification)

# Split the dataset into training and testing sets
X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification_encoded, test_size=0.2, random_state=42)

# Create a Decision Tree classifier for match result prediction
classifier_model = DecisionTreeClassifier(random_state=42)
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
home_team_input = "Man City"
away_team_input = "Man United"

# Predict the match result
predicted_result = predict_match_result(home_team_input, away_team_input)


# Split the dataset into training and testing sets for regression
X_train, X_test, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Create separate Decision Tree regressors for each goal feature
regressor_models = {}
for column in y_regression.columns:
    regressor_model = DecisionTreeRegressor(random_state=42)
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
    
def simulate_match(home_team, away_team):
    try:
        # Encode team names
        home_team_encoded = label_encoder_X.transform([home_team])[0]
        away_team_encoded = label_encoder_X.transform([away_team])[0]
        
        # Create input data for prediction
        match_data = [[home_team_encoded, away_team_encoded]]
        
        # Make predictions for goals
        goals = predict_goals(home_team, away_team)
        return goals
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
        goals = simulate_match(home_team, away_team)
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
print(f"Predicted goals for {home_team_input} vs {away_team_input}: {predicted_goals}")

overall_winner = determine_overall_winner(home_team_input, away_team_input)
print(f"Overall winner between {home_team_input} and {away_team_input}: {overall_winner}")
print(f"Predicted goals for {home_team_input} vs {away_team_input}: {predicted_goals}")
print(f"Predicted result for {home_team_input} vs {away_team_input}: {predicted_result}")