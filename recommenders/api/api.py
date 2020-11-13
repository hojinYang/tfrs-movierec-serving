import os
from flask import Flask, request, jsonify
from pathlib import Path
import sqlite3
import tensorflow as tf
from recommenders import rank, retrieve
from config import Config
from flask_sqlalchemy import SQLAlchemy
import numpy as np


DB_PATH = Path(__file__).resolve().parents[2] / "app.db"
WANDB_PATH = Path(__file__).resolve().parents[1] / "wandb"
BEST_RETR_RUN = "run-20201113_034446-boy4apok/files"
BEST_RANK_RUN = "run-20201113_102942-9xmrl22y/files"
MIN_CANDID = 100
NUM_RECOMMEND = 10 

api = Flask(__name__)
api.config.from_object(Config)
SQLAlchemy(api)

retrieval_model = retrieve.RetrievalModel(WANDB_PATH / BEST_RETR_RUN)
ranking_model = rank.RankingModel(WANDB_PATH / BEST_RANK_RUN)

@api.route("/")
def index():
    return "recommenders api"

@api.route("/v1/predict")
def predict():
    from app.models import User, Movie
    uid = request.args.get("uid")
    u = User.query.get(uid)
    history = [m.movieid for m in u.movies]
    
    u_feature = {'userid':tf.constant([u.userid])}
    candids = retrieval_model.predict(u_feature, num_candids = MIN_CANDID + len(history))

    candids = list(set(candids) - set(history))
    features = {'userid':tf.constant([u.userid]*len(candids)),
                'movieid': tf.constant(candids)
                }

    preds = ranking_model.predict(features)
    item_pred = sorted(list(zip(candids, preds)),key=lambda tup: -tup[1])[:NUM_RECOMMEND]
    out = { 'recs': [t[0] for t in item_pred],
            'preds': [str(t[1]) for t in item_pred]
            }
    
    return jsonify(out)
    

def main():
    api.run(host="0.0.0.0", port=8000, debug=False) 

if __name__ == "__main__":
    main()