import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '01-22.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Selecting relevant columns
X = df[['HomeTeam', 'AwayTeam']]
y = df['FTR']  # Only include match result

# Label encoding for 'Home Team' and 'Away Team'
label_encoder_X = LabelEncoder()
X['HomeTeam'] = label_encoder_X.fit_transform(X['HomeTeam'])
X['AwayTeam'] = label_encoder_X.transform(X['AwayTeam'])

# Encode match result 'FTR'
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier for match result prediction
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# Function to predict match result
def predict_match_result(home_team, away_team):
    try:
        # Encode team names
        home_team_encoded = label_encoder_X.transform([home_team])[0]
        away_team_encoded = label_encoder_X.transform([away_team])[0]

        # Create input data for prediction
        match_data = [[home_team_encoded, away_team_encoded]]

        # Make prediction
        prediction = model.predict(match_data)
        predicted_result = label_encoder_y.inverse_transform(prediction)[0]

        return predicted_result
    except ValueError as e:
        print(f"Error: {e}")
        return "Unknown"


# Input example: Man United vs Leeds
home_team_input = "Burnley"
away_team_input = "Bournemouth"

# Predict the result
predicted_result = predict_match_result(home_team_input, away_team_input)
print(f"Predicted result for {home_team_input} vs {away_team_input}: {predicted_result}")
