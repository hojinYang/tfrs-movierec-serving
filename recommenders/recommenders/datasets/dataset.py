import os
from pathlib import Path
import csv
import tensorflow as tf
import sqlite3
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[3] / "parsed_data"
DB_PATH = Path(__file__).resolve().parents[3] / "webserver" / "app.db"
RATING_TRAIN_FILENAME = "ratings_train.csv"
RATING_TEST_FILENAME = "ratings_test.csv"
MOVIE_FILENAME = "movies.csv"


class Dataset:
    """Simple class for datasets."""
    def __init__(self, test_fraction=0.3, batch_size=512):
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        
        self.train = None
        self.test = None
        self.movies = None
        
    def load_or_generate_data(self, update_to_latest_db=True):
        dirname = _load_or_generate_csv_data(self.test_fraction, update_to_latest_db)

        self.train = tf.data.experimental.make_csv_dataset(os.path.join(dirname, RATING_TRAIN_FILENAME),batch_size=self.batch_size,num_epochs=1)
        self.test = tf.data.experimental.make_csv_dataset(os.path.join(dirname, RATING_TEST_FILENAME),batch_size=self.batch_size,num_epochs=1)
        self.movies = tf.data.experimental.make_csv_dataset(os.path.join(dirname, MOVIE_FILENAME),batch_size=self.batch_size,num_epochs=1,shuffle=False)

    @property
    def unique_user_ids(self):
        user_ids = self.train.map(lambda x: x["userid"])
        return np.unique(np.concatenate(list(user_ids)))
        
    @property
    def unique_movie_ids(self):
        movie_ids = self.train.map(lambda x: x["movieid"])
        return np.unique(np.concatenate(list(movie_ids)))


def _load_or_generate_csv_data(test_fraction, update_to_latest_db):
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    list_of_dirs = [os.path.join(DATA_PATH, d) for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

    if len(list_of_dirs) > 0:
        latest_dir = max(list_of_dirs, key=os.path.getctime)
        if not update_to_latest_db:
            print("Loaded latest dataset(without update check)")
            return latest_dir

        if os.path.getctime(latest_dir) >= os.path.getmtime(DB_PATH):
            print("No DB update... Loaded latest dataset")
            return latest_dir

    print("Generating New dataset...")
    db_mtime = os.path.getmtime(DB_PATH)
    datadir = os.path.join(DATA_PATH, str(db_mtime))
    os.mkdir(datadir)
    
    con = sqlite3.connect(DB_PATH)
    
    with open(os.path.join(datadir,MOVIE_FILENAME), 'w') as f: 
        cursor = con.execute('select * from movie')
        outcsv = csv.writer(f)
        outcsv.writerow(x[0] for x in cursor.description)
        outcsv.writerows(cursor.fetchall())

    tr = open(os.path.join(datadir,RATING_TRAIN_FILENAME), 'w')
    te = open(os.path.join(datadir,RATING_TEST_FILENAME), 'w')

    tr_outcsv = csv.writer(tr)
    te_outcsv = csv.writer(te)
    
    cursor = con.execute('select * from user_movie_rating') #serMovieRating
    tr_outcsv.writerow(x[0] for x in cursor.description)
    te_outcsv.writerow(x[0] for x in cursor.description)
    
    for row in cursor.fetchall():
        if np.random.random_sample() > test_fraction:
            tr_outcsv.writerow(x for x in row)
        else:
            te_outcsv.writerow(x for x in row)

    return datadir

