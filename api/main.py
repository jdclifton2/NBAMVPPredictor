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
    
    """
    URL = "https://www.basketball-reference.com/leagues/NBA_2023_per_game.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    stats_table = soup.find('table', {'id': 'per_game_stats'})
    player_stats = []
    rows = stats_table.tbody.find_all('tr', attrs={'class': 'full_table'})
    for row in rows:
            player_stats.append({
            'Player': row.find('td', {'data-stat': 'player'}).text,
            'Team': row.find('td', {'data-stat': 'team_id'}).text,
            'GP': row.find('td', {'data-stat': 'blk_per_g'}).text,
            'PPG': row.find('td', {'data-stat': 'pts_per_g'}).text,
            'DRPG': row.find('td', {'data-stat': 'drb_per_g'}).text,
            'APG': row.find('td', {'data-stat': 'ast_per_g'}).text,
            'FG': row.find('td', {'data-stat': 'fg_per_g'}).text,
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


@app.post('/predict', tags=["predictions"])
async def get_prediction(mvp: MVP):
    URL = "https://www.basketball-reference.com/leagues/NBA_2022_per_game.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    stats_table = soup.find('table', {'id': 'per_game_stats'})
    player_stats = []
    for row in stats_table.tbody.find_all('tr'):
        for row2 in row.find_all('td'):
            #print(row2)
            player_stats.append({
            'Player': row.find('td', {'data-stat': 'player'}).text,
            'Team': row.find('td', {'data-stat': 'team_id'}).text,
            'GP': row.find('td', {'data-stat': 'blk_per_g'}).text,
            'PPG': row.find('td', {'data-stat': 'pts_per_g'}).text,
            'DRPG': row.find('td', {'data-stat': 'drb_per_g'}).text,
            'APG': row.find('td', {'data-stat': 'ast_per_g'}).text,
            'FG': row.find('td', {'data-stat': 'fg_per_g'}).text,
        })

    # # for row2 in row.find_all('td'):
    # #     print(row2.text)


    rf_model = joblib.load("mvp_random_forest.joblib") # ['PTSPG', 'ASTPG', 'WS', 'BLKPG', 'DRBPG', 'VORP', 'BPM', 'USG%', 'FGPG']
    scaled_log_model = joblib.load("mvp_scaled_log.joblib")
    data = dict(mvp)['data']
    #print(model)
    prediction = rf_model.predict(data).tolist()
    log_proba = rf_model.predict_proba(data).tolist()
    return {"prediction": prediction,
            "probs": log_proba,
            "mvp_proba": log_proba[0][1],
            "non_mvp_proba": log_proba[0][0]}