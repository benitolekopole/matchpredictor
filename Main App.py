import requests
import streamlit as st
import pandas as pd
import numpy as np

def fetch_football_data(api_key, league_code='PL'):
    """
    Fetches finished match results for a specific league.
    League Codes: 'PL' (Premier League), 'PD' (La liga), 'BL1 (Bundesliga) 
    
    """
url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
headers = {'X-Auth-Token':api_key}
params = {'status':'FINISHED'} #We only want past matches to calculate performance
try:
    response = request.get(url, headers=headers, params=params)
    response.raise_for_status() #Check for errors
    data = response.json()
    matches =[]
    for match in data['matches']:
        matches.append({
            'Date': match['utcDate'],
            'HomeTeam': match['homeTeam']['name'],
            'AwayTeam': match['awayTeam']['name'],
            'HomeGoals': match['score']['fullTime']['home'],
            'AwayGoals': match['score']['fullTime']['away'],
        
        })
    return pd.DataFrame(matches)
except Exception as e:
    st.error(f"Error fetching data:{e}")
    return None

# Use caching so we only fetch data once every 12hours
@st.cache_data(ttl=43200)
def get_cached_data(api_key):
    return fetch_football_data(api_key)

API_KEY = st.secrets["FOOTBALL_API_KEY"]
df = get_cached_data(API_KEY)

if df is not None:
    st.success(f"Successfully loaded {len(df)} matches!")

# Set up the web page
st.set_page_config(page_title="Pro Football Predictor", layout"wide")
st.title("⚽ European Football Match Predictor")

def calculate_team_stats(df):
    # Calculate league averages
    avg_home_goals = df['HomeGoals'].mean()
    avg_away_goals = df['AwayGoals'].mean()

    #Home Attack Strength = (Team's Home Goals / Home Matches) / League Avg Home Goals
    home_stats = df.groupby('HomeTeam').agg({'HomeGoals':'mean', 'AwayGoals':'mean'})
    home_stats.columns = ['Att_Home', 'Def_Home']
    home_stats['Att_Home'] /= avg_home_goals
    home_stats['Def_Home'] /= avg_away_goals 

    #Away Attack Strength = (Team's Away Goals / Away Matches) / League Avg Away Goals
    away_stats = df.groupby('AwayTeam').agg({'AwayGoals':'mean', 'HomeGoals':'mean'})
    away_stats.columns = ['Att_Away', 'Def_Away']
    away_stats['Att_Away'] /= avg_away_goals
    away_stats['Def_Away'] /= avg_home_goals 

    return home_stats, away_stats, avg_home_goals, avg_away_goals

    def predict_match(home_team, away_team, home_stats, away_stats, avg_h, avg_a):
        #Expected goals for Home Team
        home_exp = home_stats.loc[home_team, 'Att_Home'] * away_stats.loc[away_team, 'Def_Away'] * avg_h

        #Expected goals for Away Team
        away_exp = away_stats.loc[away_team, 'Att_Away'] * home_stats.loc[home_team, 'Def_Home'] * avg_a

        #Calaculate probability matrix (up to 5 goals each)
        matrix = np.outer(poisson.pmf(range(6), home_exp), poisson.pmf(range(6),away_exp))

        # Win/Draw/Loss Probabilities
        prob_home = np.sum(np.tril(matrix, -1))
        prob_draw = np.sum(np.diag(matrix))
        prob_away = np.sum(np.triu(matrix, 1))

        # Most likely score
        most_likely_score = np.unravel_index(np.argmax(matrix), matrix.shape)

        return prob_home, prob_draw, prob_away, most_likely_score

        #Assuming 'data.csv' has columns:HomeTeam, AwayTeam, HomeGoals, AwayGoals
        df = pd.read_csv('match_history.csv')
        h_stats, a_stats, avg_h, avg_a =calculate_team_stats(df)

        col1, col2 = st.columns(2)
        with col1: home_choice = st.selectbox("Select Home Team", h_stats.index)
        with col2: away_choice = st.selectbox("Select Away Team", a_stats.index)

        if st.button("Predict Outcome"):p_h, p_d, p_a, score = predict_match(home_choice, away_choice, h_stats, a_stats, avg_h, avg_a)

        st.write(f"### Predicted Score: {score[0]} - {score[1]}")
        st.write(f"** Win Probability : ** {home_choice} : {p_h: .1%}, Draw: {p_d: .1%}, {away_choice}: {p_a: .1%}")

