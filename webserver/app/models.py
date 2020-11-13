from app import db


class UserMovieRating(db.Model):
    userid = db.Column(db.Integer, db.ForeignKey('user.userid'), primary_key=True)
    movieid = db.Column(db.Integer, db.ForeignKey('movie.movieid'), primary_key=True)
    rating = db.Column(db.Float)
    user = db.relationship('User', back_populates='movies')
    movie = db.relationship('Movie', back_populates='users')

    def __repr__(self):
        return '<User {}, Movie {}, Rating {}>'.format(self.user.username, self.movie.title, self.rating)

class User(db.Model):
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    movies = db.relationship('UserMovieRating', back_populates='user')

    def __repr__(self):
        return '<User {}>'.format(self.username)    

    def is_watched(self, movie):
        for r in self.movies:
            if r.movieid == movie.movieid:
                return r
        return None

class Movie(db.Model):
    movieid = db.Column(db.Integer, primary_key=True)
    movielensid = db.Column(db.Integer, index=True, unique=True) 
    imdbid = db.Column(db.Integer, index=True, unique=True)
    title = db.Column(db.String(64), index=True)
    cover = db.Column(db.String)
    plot = db.Column(db.String)
    year = db.Column(db.Integer)
    users = db.relationship('UserMovieRating', back_populates='movie')

    def __repr__(self):
        return '<Movie {}>'.format(self.title)   
