from fastapi import FastAPI
import sklearn
import joblib
from typing import List 
from pydantic import BaseModel, conlist
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
 
app = FastAPI(title="MVP Predictor API", description="API for predicting NBA MVP", version="1.0")

class MVP(BaseModel):
    data: List[conlist(float, min_items=5, max_items=5)] # VORP WS BPM OWS WS/48
    #jokic 2022
    # [ 
    #   9.8,
    #   15.2,
    #   13.7,
    #   10.8,
    #   0.296
    # ]

# @app.on_event('startup')
# def load_model():
#     model = joblib.load("adaboost.joblib")
# # load the model from a file

def get_data() -> pd.DataFrame:
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

    df = pd.merge(stats_df, adv_df, on=['Player', 'Team'], how="inner")
    return df 


@app.get('/predict', tags=["predictions"])
async def get_prediction():

    stats_df = get_data()
    players = stats_df['Player']
    teams = stats_df['Team']
    X = stats_df.drop(columns=['Player', 'Team'])[['PTSPG', 'ASTPG', 'WS', 'BLKPG', 'DRBPG', 'VORP', 'BPM', 'USG%', 'FGPG']]

    rf_model = joblib.load("mvp_random_forest.joblib") # ['PTSPG', 'ASTPG', 'WS', 'BLKPG', 'DRBPG', 'VORP', 'BPM', 'USG%', 'FGPG']
    scaled_log_model = joblib.load("mvp_scaled_log.joblib")
    #data = dict(mvp)['data']
    #print(model)
    #prediction = rf_model.predict(data).tolist()
    #log_proba = rf_model.predict_proba(data).tolist()

    proba_scaled_log = scaled_log_model.predict_proba(X).tolist()
    proba_RF = rf_model.predict_proba(X).tolist()
    mvp_log_probs = []
    mvp_rf_probs = []
    for prob in proba_scaled_log:
        mvp_log_probs.append(prob[1])
    for prob in proba_RF:
        mvp_rf_probs.append(prob[1])
    # print(hand_select_models['Scaledlogistic']['fitted_model'].predict_proba(X).tolist())
    # print(proba_RF)
    # stats_df['results_scaled_log'] = results_scaled_log
    # stats_df['results_RF'] = results_RF
    stats_df['proba_scaled_log'] = mvp_log_probs
    stats_df['proba_RF'] = mvp_rf_probs
    result_df = stats_df.sort_values(by='proba_RF', ascending=False)[['Player', 'proba_scaled_log', 'proba_RF']].head(10)
    return result_df.to_dict()
    # return {"prediction": prediction,
    #         "probs": log_proba,
    #         "mvp_proba": log_proba[0][1],
    #         "non_mvp_proba": log_proba[0][0]}