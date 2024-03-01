import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '15-22.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Selecting relevant columns
X = df[['HomeTeam', 'AwayTeam']]
y_classification = df['FTR']  # Match result
y_regression = df[['FTHG', 'FTAG', 'HTHG', 'HTAG']]  # Goals

# Label encoding for 'Home Team' and 'Away Team'
label_encoder_X = LabelEncoder()
X['HomeTeam'] = label_encoder_X.fit_transform(X['HomeTeam'])
X['AwayTeam'] = label_encoder_X.transform(X['AwayTeam'])

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
home_team_input = "Burnley"
away_team_input = "Bournemouth"

# Predict the match result
predicted_result = predict_match_result(home_team_input, away_team_input)
print(f"Predicted result for {home_team_input} vs {away_team_input}: {predicted_result}")

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
            predictions[column] = prediction[0]
        
        return predictions
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Predict goals
predicted_goals = predict_goals(home_team_input, away_team_input)
print(f"Predicted goals for {home_team_input} vs {away_team_input}: {predicted_goals}")
