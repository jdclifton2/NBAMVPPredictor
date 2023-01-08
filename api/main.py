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


@app.post('/predict', tags=["predictions"])
async def get_prediction(mvp: MVP):
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