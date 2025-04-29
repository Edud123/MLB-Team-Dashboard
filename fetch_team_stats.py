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
import tensorflow as tf
import joblib

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
        batting_data = {}  # Initialize dictionary to collect batting data
        for row in rows:
            cols = row.find_all('td')
            team_name_col = row.find('th')  # Team name is often in the <th> tag
            if cols and team_name_col:
                team_name = team_name_col.text.strip()
                obp = float(cols[17].text.strip()) if len(cols) > 17 else 0.0  # Column index for OBP
                slg = float(cols[18].text.strip()) if len(cols) > 18 else 0.0  # Column index for SLG

                # Collect refined data
                batting_data[team_name] = {'obp': obp, 'slg': slg}

        # Print the original scraped batting data for debugging
        print(":) Debug: Original Scraped Batting Data")
        print(batting_data)

        return batting_data

    except requests.exceptions.RequestException as e:
        print(f"Error during scraping: {e}")
        return {}

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

# Load 2025 ERA data from 'Team ERA 2025.csv'
def load_pitching_stats(file_path):
    print(":) Loading 2025 pitching stats from CSV...")
    pitching_data = {}

    # Handle BOM in CSV headers
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        print("CSV Headers:", reader.fieldnames)  # Print the headers for debugging

    # Reload the file to process rows
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            team_name = row['Team']  # Use full-length team names directly
            era = float(row['ERA'])
            pitching_data[team_name] = {'era': era}
    return pitching_data

# Scale the ERA data in 'Team ERA 2025.csv'
def scale_era_data(file_path):
    print(":) Scaling ERA data...")
    df = pd.read_csv(file_path)

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Scale the ERA column
    df['ERA'] = scaler.fit_transform(df[['ERA']])

    # Save the scaled data back to the CSV
    df.to_csv(file_path, index=False)
    print(":) Scaled ERA data saved to", file_path)

# Scale the ERA data in 'Team Important Stats 2010-2025.csv'
def scale_era_in_important_stats(file_path):
    print(":) Scaling ERA data in Team Important Stats...")
    df = pd.read_csv(file_path)

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Scale the ERA column
    df['ERA'] = scaler.fit_transform(df[['ERA']])

    # Save the scaled data back to the CSV
    df.to_csv(file_path, index=False)
    print(":) Scaled ERA data saved to", file_path)

def normalize_team_name(name):
    # Map full team names to their abbreviations
    team_name_mapping = {
        'Arizona Diamondbacks': 'ari',
        'Athletics': 'ath',
        'Atlanta Braves': 'atl',
        'Baltimore Orioles': 'bal',
        'Boston Red Sox': 'bos',
        'Chicago Cubs': 'chc',
        'Chicago White Sox': 'chw',
        'Cincinnati Reds': 'cin',
        'Cleveland Guardians': 'cle',
        'Colorado Rockies': 'col',
        'Detroit Tigers': 'det',
        'Houston Astros': 'hou',
        'Kansas City Royals': 'kcr',
        'Los Angeles Angels': 'laa',
        'Los Angeles Dodgers': 'lad',
        'Miami Marlins': 'mia',
        'Milwaukee Brewers': 'mil',
        'Minnesota Twins': 'min',
        'New York Mets': 'nym',
        'New York Yankees': 'nyy',
        'Philadelphia Phillies': 'phi',
        'Pittsburgh Pirates': 'pit',
        'San Diego Padres': 'sdp',
        'Seattle Mariners': 'sea',
        'San Francisco Giants': 'sfg',
        'St. Louis Cardinals': 'stl',
        'Tampa Bay Rays': 'tbr',
        'Texas Rangers': 'tex',
        'Toronto Blue Jays': 'tor',
        'Washington Nationals': 'wsn'
    }
    return team_name_mapping.get(name.strip(), '').lower()

# Remove normalization for ERA data and use full-length team names consistently
def predict_win_rates(obp_slg_data, era_data):
    # Filter out 'League Average' and blank entries from OBP/SLG data
    obp_slg_data = {team: stats for team, stats in obp_slg_data.items() if team != 'League Average' and team.strip() != ''}

    # Remove scaling for OBP and SLG data
    for team, stats in obp_slg_data.items():
        stats['obp'] = stats['obp']  # Keep OBP as is
        stats['slg'] = stats['slg']  # Keep SLG as is

    # Print all scraped data for debugging
    print(":) Debug: Scraped OBP/SLG Data")
    print(obp_slg_data)

    print(":) Debug: Scraped ERA Data")
    print(era_data)

    # Merge OBP, SLG, and ERA into a feature set
    features = []
    teams_with_data = []  # Track teams with complete data

    for team, stats in obp_slg_data.items():
        if team in era_data:  # Use full-length team names directly
            obp = stats.get('obp', None)
            slg = stats.get('slg', None)
            era = era_data[team].get('era', None)

            if obp is not None and slg is not None and era is not None:
                features.append([obp, slg, era])
                teams_with_data.append(team)

    if not features:
        print("Debug: OBP/SLG Data:", obp_slg_data)
        print("Debug: ERA Data:", era_data)
        raise ValueError("No valid data found for prediction. Check input data.")

    # Convert features to DataFrame with correct column names
    features_df = pd.DataFrame(features, columns=['OBP', 'SLG', 'ERA'])

    # Scale the features
    features_scaled = scaler.transform(features_df)

    # Predict win rates
    predicted_win_rates = model.predict(features_scaled)

    # Map predictions back to team names
    predictions = {}
    for i, team in enumerate(teams_with_data):
        predictions[team] = predicted_win_rates[i][0]

    return predictions

# Normalize team names in the Streamlit app
if __name__ == "__main__":
    st.title("MLB Team Dashboard")
    st.subheader("Predict Game Winner")

    st.write("""
    This dashboard uses a machine learning model trained on historical MLB data (2010-2025) to predict game winners. 
    The model leverages key statistics such as On-Base Percentage (OBP), Slugging Percentage (SLG), and Earned Run Average (ERA). 
    Real-time batting data is fetched from Baseball Reference, and pitching data is sourced from the latest available stats to make game-winner predictions.
    """)

    # Load the pre-trained model with the correct loss function
    model = tf.keras.models.load_model('win_rate_predictor_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

    # Load the scaler used during training
    scaler = joblib.load('scaler.pkl')

    # Scale the ERA data
    scale_era_data("Team ERA 2025.csv")
    scale_era_in_important_stats("Team Important Stats 2010-2025.csv")

    # Fetch OBP and SLG from the API
    obp_slg_data = scrape_baseball_reference()

    # Load 2025 ERA data
    era_data = load_pitching_stats("Team ERA 2025.csv")

    # Print all predictions for debugging
    print("Debug: All Predictions")
    predictions = predict_win_rates(obp_slg_data, era_data)
    for team, win_rate in predictions.items():
        print(f"{team}: {win_rate:.2f}")

    # Team selection dropdowns
    teams = list(obp_slg_data.keys())
    home_team = st.selectbox("Select Home Team", teams)
    away_team = st.selectbox("Select Away Team", teams)

    if st.button("Submit"):
        # Get predictions for the selected teams
        home_team_win_rate = round(predictions.get(home_team, 0) * 100)
        away_team_win_rate = round(predictions.get(away_team, 0) * 100)

        # Determine the predicted winner
        if home_team_win_rate > away_team_win_rate:
            st.success(f"🎉 {home_team} is predicted to win! With a predicted season win rate of {home_team_win_rate}% 🏆")
        elif away_team_win_rate > home_team_win_rate:
            st.success(f"🎉 {away_team} is predicted to win! With a predicted season win rate of {away_team_win_rate}% 🏆")
        else:
            st.info("It's a tie! Both teams have the same predicted win rate.")

        # Display the winning team's logo only
        if home_team_win_rate > away_team_win_rate:
            winning_logo_path = f"logos/{home_team.lower().replace(' ', '_')}.svg"
        else:
            winning_logo_path = f"logos/{away_team.lower().replace(' ', '_')}.svg"

        st.image(winning_logo_path, width=150)