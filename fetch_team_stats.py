from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import streamlit as st

def scrape_baseball_reference():
    url = "https://www.baseball-reference.com/leagues/MLB/2025.shtml"
    print(":) Scraping Baseball Reference for advanced stats...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Scrape team stats table (adjust selectors as needed)
        team_stats_table = soup.find('table', {'id': 'teams_standard_batting'})
        if not team_stats_table:
            print(":( Could not find the team stats table.")
            return

        # Extract data from the table
        rows = team_stats_table.find_all('tr')
        batting_data = []  # Initialize list to collect batting data
        for row in rows:
            cols = row.find_all('td')
            team_name_col = row.find('th')  # Team name is often in the <th> tag
            if cols and team_name_col:
                team_name = team_name_col.text.strip()
                obp = cols[10].text.strip() if len(cols) > 10 else "N/A"  # Example column for OBP
                slg = cols[11].text.strip() if len(cols) > 11 else "N/A"  # Example column for SLG

                # Collect refined data
                batting_data.append({'name': team_name, 'obp': obp, 'slg': slg})

        # Example: Scrape pitching stats table for ERA and other metrics
        pitching_stats_table = soup.find('table', {'id': 'teams_standard_pitching'})
        if pitching_stats_table:
            # Debugging: Print the entire HTML to check for pitching stats table
            print(":) Debugging: Printing HTML to check for pitching stats table...")
            print(soup.prettify())

            print(":) Found pitching stats table. Extracting ERA...")
            rows = pitching_stats_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                team_name_col = row.find('th')  # Team name is often in the <th> tag
                if cols and team_name_col:
                    team_name = team_name_col.text.strip()
                    era = cols[8].text.strip() if len(cols) > 8 else "N/A"  # Example column for ERA
                    print(f"Team: {team_name}, ERA: {era}")

    except requests.exceptions.RequestException as e:
        print(f"Error during scraping: {e}")
    
    return batting_data  # Return the collected batting data

def scrape_mlb_standings():
    url = "https://www.mlb.com/standings/"
    print(":) Scraping MLB Standings for wins and losses...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Scrape standings table for wins and losses
        standings_table = soup.find('table')  # Adjust selector based on MLB standings page structure
        if not standings_table:
            print("Could not find the standings table on MLB.com.")
            return

        rows = standings_table.find_all('tr')
        standings_data = []  # Initialize list to collect standings data
        for row in rows:
            cols = row.find_all('td')
            team_name_col = row.find('th')  # Team name is often in the <th> tag
            if cols and team_name_col:
                team_name = team_name_col.text.strip()
                # Refine logic to correctly extract wins and losses based on debugging output
                try:
                    wins = cols[0].text.strip() if len(cols) > 0 else "N/A"  # Adjusted column index for wins
                    losses = cols[1].text.strip() if len(cols) > 1 else "N/A"  # Adjusted column index for losses
                except ValueError:
                    wins = "N/A"
                    losses = "N/A"

                # Collect refined data
                standings_data.append({'name': team_name, 'wins': wins, 'losses': losses})

    except requests.exceptions.RequestException as e:
        print(f"Error during scraping MLB Standings: {e}")
    
    return standings_data  # Return the collected standings data

def load_pitching_stats(file_path):
    print(":) Loading pitching stats from CSV...")
    pitching_data = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            team_name = row['Team']
            era = row['ERA']
            pitching_data[team_name] = {'era': era}
    return pitching_data

def load_team_stats(file_path):
    print(":) Loading team stats from CSV...")
    team_data = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            team_name = row['Team']
            war = row['WAR']
            team_data[team_name] = {'war': war}
    return team_data

def normalize_team_name(name):
    # Normalize team names to ensure consistency across datasets
    return name.strip().lower().replace(" ", "_")

def merge_tables(standings_data, batting_data, pitching_data, team_data):
    print(":) Merging tables...")

    # Normalize team names for quick lookup
    standings_dict = {normalize_team_name(team['name']): team for team in standings_data}
    batting_dict = {normalize_team_name(team['name']): team for team in batting_data}
    pitching_dict = {normalize_team_name(team): data for team, data in pitching_data.items()}
    team_dict = {normalize_team_name(team): data for team, data in team_data.items()}

    # Merge data where normalized team names match
    merged_data = []
    for team_name in standings_dict:
        if team_name in batting_dict and team_name in pitching_dict and team_name in team_dict:
            # Calculate win rate
            try:
                wins = int(standings_dict[team_name]['wins'])
                losses = int(standings_dict[team_name]['losses'])
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            except ValueError:
                win_rate = "N/A"

            # Ensure wins and losses are added as integers
            merged_row = {
                'name': team_name.replace("_", " ").title(),
                'wins': int(standings_dict[team_name]['wins']),
                'losses': int(standings_dict[team_name]['losses']),
                'win_rate': win_rate,
                'obp': float(batting_dict[team_name]['obp']),
                'slg': float(batting_dict[team_name]['slg']),
                'era': float(pitching_dict[team_name]['era']),
                'war': float(team_dict[team_name]['war'])
            }
            merged_data.append(merged_row)

    print(":) Merged data:", merged_data)
    print(":) Debugging: Merged data keys:", merged_data[0].keys() if merged_data else "No data")
    return merged_data

def clean_and_scale_data(merged_data):
    print(":) Cleaning and scaling data for ML model...")

    # Convert merged data to a DataFrame
    df = pd.DataFrame(merged_data)

    # Drop rows with missing or invalid values
    df.replace("N/A", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert numerical columns to appropriate types
    df["wins"] = df["wins"].astype(int)
    df["losses"] = df["losses"].astype(int)
    df["win_rate"] = df["win_rate"].astype(float)
    df["obp"] = df["obp"].astype(float)
    df["slg"] = df["slg"].astype(float)
    df["era"] = df["era"].astype(float)

    # Scale numerical features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[["obp", "slg", "era"]])
    df[["obp", "slg", "era"]] = scaled_features

    # Save cleaned and scaled data to a new CSV file
    output_file = "cleaned_scaled_data.csv"
    df.to_csv(output_file, index=False)
    print(f":) Cleaned and scaled data saved to {output_file}")

    return df

def build_and_evaluate_models(cleaned_data):
    print(":) Building and evaluating Neural Network model...")

    # Define features (X) and target (y)
    X = cleaned_data[["obp", "slg", "era"]]
    y = cleaned_data["win_rate"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Neural Network Model
    nn_model = MLPRegressor(random_state=42, max_iter=500)
    nn_model.fit(X_train, y_train)
    nn_y_pred = nn_model.predict(X_test)
    nn_mse = mean_squared_error(y_test, nn_y_pred)
    nn_r2 = r2_score(y_test, nn_y_pred)
    evaluate_with_percentage_error(y_test, nn_y_pred, "Neural Network")

    return nn_model

def improve_models(cleaned_data):
    print(":) Improving models with hyperparameter tuning and feature engineering...")

    # Define features (X) and target (y)
    X = cleaned_data[["obp", "slg", "era"]]
    y = cleaned_data["win_rate"]

    # Add polynomial features to capture non-linear relationships
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Neural Network with hyperparameter tuning
    nn_model = MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, max_iter=1000, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_y_pred = nn_model.predict(X_test)
    nn_mse = mean_squared_error(y_test, nn_y_pred)
    nn_r2 = r2_score(y_test, nn_y_pred)
    print(f":) Improved Neural Network Evaluation:\nMean Squared Error: {nn_mse}\nR-squared: {nn_r2}")

    # Cross-validation for Neural Network
    nn_cv_scores = cross_val_score(nn_model, X_poly, y, cv=5, scoring='r2')
    print(f":) Neural Network Cross-Validation R-squared: {nn_cv_scores.mean()} (+/- {nn_cv_scores.std()})")

    return nn_model

def evaluate_with_percentage_error(y_test, y_pred, model_name):
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f":) {model_name} Evaluation:\nMean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

def display_dashboard(cleaned_data):
    # Update the dashboard to include dropdowns for team comparison
    st.title("MLB Team Dashboard")
    st.header("Predict Game Winner")

    # Define features (X) for prediction
    X = cleaned_data[["obp", "slg", "era"]]

    # Load the trained Neural Network model
    nn_model = MLPRegressor(random_state=42, max_iter=500)
    nn_model.fit(X, cleaned_data["win_rate"])

    # Predict win rates
    predicted_win_rates = nn_model.predict(X)
    cleaned_data["predicted_win_rate"] = predicted_win_rates

    # Dropdowns for team selection
    team_names = cleaned_data["name"].tolist()
    team1 = st.selectbox("Select Home Team", team_names)
    team2 = st.selectbox("Select Away Team", team_names)

    # Submit button to determine the winner
    if st.button("Submit"):
        team1_data = cleaned_data[cleaned_data["name"] == team1]
        team2_data = cleaned_data[cleaned_data["name"] == team2]

        team1_win_rate = team1_data["predicted_win_rate"].values[0]
        team2_win_rate = team2_data["predicted_win_rate"].values[0]

        if team1_win_rate > team2_win_rate:
            st.success(f"🎉 {team1} is predicted to win! With a predicted win rate of {team1_win_rate:.2f}! 🏆")
            logo_path = f"logos/{team1.replace(' ', '_').lower()}.svg"
        else:
            st.success(f"🎉 {team2} is predicted to win! With a predicted win rate of {team2_win_rate:.2f}! 🏆")
            logo_path = f"logos/{team2.replace(' ', '_').lower()}.svg"

        # Display the winning team's logo
        st.image(logo_path, use_container_width=True)  # Updated to use 'use_container_width' instead of 'use_column_width'

if __name__ == "__main__":
    # Collect data from both sources
    standings_data = scrape_mlb_standings()
    batting_data = scrape_baseball_reference()
    pitching_data = load_pitching_stats("Team Pitching Stats 2010-2025.csv")
    team_data = load_team_stats("Team stats 2010-2025.csv")

    # Merge the tables
    merged_table = merge_tables(standings_data, batting_data, pitching_data, team_data)

    # Clean and scale data for ML model
    cleaned_data = clean_and_scale_data(merged_table)

    # Build and evaluate Neural Network model
    nn_model = build_and_evaluate_models(cleaned_data)

    # Improve models with hyperparameter tuning and feature engineering
    improved_nn_model = improve_models(cleaned_data)

    # Display the dashboard using Streamlit
    display_dashboard(cleaned_data)