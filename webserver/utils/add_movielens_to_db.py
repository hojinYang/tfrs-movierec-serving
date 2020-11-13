import sys
from pathlib import Path
from imdb import IMDb
from app import db
from app.models import Movie, User, UserMovieRating

DATA_DIRNAME = Path(__file__).resolve().parents[2] / 'data' / 'ml-latest-small'
print(DATA_DIRNAME)

ia = IMDb()
links = DATA_DIRNAME /'links.csv'
ratings = DATA_DIRNAME/ 'ratings.csv'

with open(links,'r') as f:
    line = f.readline().strip()
    n = 0
    while True:
        n+=1
        line = f.readline().strip()
        if not line:    
            break

        movielens, imdb, tmdb = line.split(',')
        movie = ia.get_movie(imdb)

        title = movie.get('title','')
        cover = movie.get('cover url','')
        year = movie.get('year',0)
        plot = movie.get('plot outline')
        m = Movie(imdbid = int(imdb),movielensid = int(movielens),\
            title = title, cover = cover, plot = plot, year = year)
        db.session.add(m)
        if n % 100 == 0:
            print(n, m)
            db.session.commit()

db.session.commit()

with open(ratings, 'r') as f:
    line = f.readline().strip()
    n = 0
    while True:
        n+=1
        line = f.readline().strip()
        if not line:
            break
        username, movielensid, rating, _ = line.split(',')
        user = User.query.filter_by(username = username).first()
        movie = Movie.query.filter_by(movielensid = int(movielensid)).first()
        if movie is None:
            continue
        if user is None:
            user = User(username=username)
            db.session.add(user)
        r = UserMovieRating(movieid = movie.movieid, userid = user.userid, rating = float(rating))
        # r.Movie = movie
        user.movies.append(r)
        movie.users.append(r)
        db.session.add(r)
        

        if n % 100 == 0:
            print(n, r)

db.session.commit()


        

        




