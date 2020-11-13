from flask import render_template, redirect, url_for
import requests
import json
from app import app, db
from app.forms import LoginForm, RatingForm
from app.models import User, Movie, UserMovieRating

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        return redirect(url_for('user', username=form.username.data))
    return render_template('index.html', title='Sign In', form=form)

@app.route('/user/<username>', methods=['GET'])
def user(username):
    form = RatingForm()
    user = User.query.filter_by(username=username).first_or_404()
    ratings = user.movies 
    recommended_movies = _get_recommendations(user.userid)
    return render_template('user.html', user=user, ratings=ratings, recs=recommended_movies, form=form)

RECOMMENDER_API_URL = 'http://0.0.0.0:8000/v1/predict'
def _get_recommendations(userid):
    res = json.loads(requests.get(RECOMMENDER_API_URL, params={'uid': userid}).content)
    recommended_movies = [{'movie':Movie.query.get(m), 'pred': p} 
                            for m, p in zip(res['recs'],res['preds'])]
    return recommended_movies
    
@app.route('/update/<username>', methods=['POST'])
def update(username):
    form = RatingForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=username).first_or_404()
        print(form.movie_id.data)
        movie = Movie.query.get(form.movie_id.data)
        rating = float(form.rating.data)
        
        r = user.is_watched(movie)
        if r is not None:
            db.session.delete(r) 
            db.session.commit()

        if rating == -1:
            return redirect(url_for('user', username=username))

        r = UserMovieRating(movieid = movie.movieid, userid = user.userid, rating = float(rating))
        user.movies.append(r)
        movie.users.append(r)
        db.session.add(r)
        db.session.commit()

        print('movie {}, user {}, rating {}'.format(movie.movieid, username, form.rating.data))
        return redirect(url_for('user', username=username))

@app.route('/user/<username>/<movieid>', methods=['GET'])
def movie(username, movieid):
    user = User.query.filter_by(username=username).first_or_404()
    movie = Movie.query.get(movieid)
    return render_template('movie.html', user=user, movie=movie)