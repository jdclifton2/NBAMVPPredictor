import sklearn
import joblib
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import gradio as gr


#rf_model = joblib.load("api/mvp_random_forest.joblib") # ['PTSPG', 'ASTPG', 'WS', 'BLKPG', 'DRBPG', 'VORP', 'BPM', 'USG%', 'FGPG']
scaled_log_model = joblib.load("api/mvp_scaled_log.joblib")
#gb_model = joblib.load("api/mvp_gb.joblib")

BANNER_URL = "https://cdn.freebiesupply.com/images/large/2x/nba-logo-transparent.png"


def get_data():
    """
    This function gets data from basketball reference and returns it as a data frame in the form of 
    Player Team GP	PPG	DRPG APG FG	WS USG% VORP BPM
    :return df: A data frame in the form of Player Team GP	PPG	DRPG APG FG	WS USG% VORP BPM
    """
    URL = "https://www.basketball-reference.com/leagues/NBA_2023_per_game.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    stats_table = soup.find('table', {'id': 'per_game_stats'})
    player_stats = []
    rows = stats_table.tbody.find_all('tr', attrs={'class': 'full_table'})
    for row in rows:
            player_stats.append({
            'Player': row.find('td', {'data-stat': 'player'}).text, # ['PTSPG', 'ASTPG', 'WS', 'BLKPG', 'DRBPG', 'VORP', 'BPM', 'USG%', 'FGPG']
            'Team': row.find('td', {'data-stat': 'team_id'}).text,
            'PTSPG': row.find('td', {'data-stat': 'pts_per_g'}).text,
            'ASTPG': row.find('td', {'data-stat': 'ast_per_g'}).text,
            'BLKPG': row.find('td', {'data-stat': 'blk_per_g'}).text,
            'DRBPG': row.find('td', {'data-stat': 'drb_per_g'}).text,
            'FGPG': row.find('td', {'data-stat': 'fg_per_g'}).text,
            })
        
    stats_df= pd.DataFrame(player_stats)

    URL = "https://www.basketball-reference.com/leagues/NBA_2023_advanced.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    stats_table = soup.find('table', {'id': 'advanced_stats'})
    rows = stats_table.tbody.find_all('tr', attrs={'class': 'full_table'})
    player_stats = []
    for row in rows:
        player_stats.append({
        'Player': row.find('td', {'data-stat': 'player'}).text,
        'Team': row.find('td', {'data-stat': 'team_id'}).text,
        'WS': row.find('td', {'data-stat': 'ws'}).text,
        'USG%': row.find('td', {'data-stat': 'usg_pct'}).text,
        'VORP': row.find('td', {'data-stat': 'vorp'}).text,
        'BPM': row.find('td', {'data-stat': 'bpm'}).text,
        })
        

    adv_df = pd.DataFrame(player_stats)

    df = pd.merge(stats_df, adv_df, on=['Player', 'Team'], how="inner").sort_values(by='Player', ascending=True)
    return df 


def get_prediction():

    stats_df = get_data()
    players = stats_df['Player']
    teams = stats_df['Team']
    X = stats_df.drop(columns=['Player', 'Team'])[['PTSPG', 'ASTPG', 'WS', 'BLKPG', 'DRBPG', 'VORP', 'BPM', 'USG%', 'FGPG']]

    proba_scaled_log = scaled_log_model.predict_proba(X).tolist()
    mvp_log_probs = []

    for prob in proba_scaled_log:
        mvp_log_probs.append(prob[1])


    stats_df['proba_scaled_log'] = mvp_log_probs

    result_df = stats_df.sort_values(by='proba_scaled_log', ascending=False)[['Player', 'proba_scaled_log']].head(10)
    return result_df.to_dict()


def respond():
    df = pd.DataFrame(get_prediction())
    return df


with gr.Blocks(title="MVP Predictor Demo") as demo:
    banner = gr.HTML(f'<img src="{BANNER_URL}" width="220" height="17">')

    with gr.Tab("Stats"):
        stats_df = gr.Dataframe(label="NBA Player Stats", type="array")
        get_stats_btn = gr.Button(value="Submit", size="lg", scale=3, variant="primary")

        get_stats_btn.click(fn=get_data, inputs=[], outputs=[stats_df])

    with gr.Tab("MVP Prediction"):  
        mvp_results = gr.Dataframe(label="MVP Predictions", type="array", headers=['Player', 'proba_scaled_log'],  datatype=["str", "number"])

        submit_btn = gr.Button(value="Submit", size="lg", scale=3, variant="primary")

        submit_btn.click(fn=respond, inputs=[], outputs=[mvp_results])
    




demo.launch()