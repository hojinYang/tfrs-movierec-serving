from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, HiddenField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    submit = SubmitField('Sign In')

class EmptyForm(FlaskForm):
    submit = SubmitField('Back')

RATINGS = [(-1,'---'),(0.5,'0.5'),(1.0,'1.0'),(1.5,'1.5'),(2.0,'2.0'),(2.5,'2.5'),(3.0,'3.0'),(3.5,'3.5'),(4.0,'4.0'),(4.5,'4.5'),(5.0,'5.0')]
class RatingForm(FlaskForm):
    movie_id = HiddenField()
    rating = SelectField('Rating', choices=RATINGS)
    submit = SubmitField('rate')

    def set_default_value(self, default):
        self.rating.default = default
        self.process()
        return ''