from flask import Flask,render_template,session,redirect,url_for,request
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,IntegerField,TextAreaField
from wtforms.validators import DataRequired
import pickle
import numpy as np
from random import shuffle
import os
import nltk


pickle_in = open("./pickle/stopword_english.pickle","rb")
stopword_english = pickle.load(pickle_in)

pickle_in = open("./pickle/wordDict.pickle","rb")
wordDict = pickle.load(pickle_in)

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [t for t in tokens if t not in stopword_english] # remove stopwords
    return tokens

basedir = os.path.abspath(os.path.dirname(__file__))

#init app
app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

class InfoForm(FlaskForm):

    id = IntegerField("Please enter your id: ",validators = [DataRequired()])
    name = StringField("Enter your name: ",validators = [DataRequired()])
    submit = SubmitField('Submit')

class MovieRating(FlaskForm):

    comment = TextAreaField("Enter your comment: ",validators = [DataRequired()])
    submit = SubmitField('Submit')


@app.route('/',methods=['GET','POST'])
def UserlogIn():
    form = InfoForm()

    if form.validate_on_submit():
        session['id'] = form.id.data
        session['name'] = form.name.data
        if(session['id']>10):
            return redirect(url_for('UserlogIn'))
        return redirect(url_for('thank'))

    return render_template('index.html',form=form)

@app.route('/home.html')
def thank():
    pickle_in = open("./pickle/popularMovie.pickle","rb")
    popularMovie = pickle.load(pickle_in)
    # shuffle(popularMovie)
    return render_template('home.html',popularMovie=popularMovie[:10])

@app.route('/recommendation.html/<name>')
def recommendation(name):
    value1 = os.path.join(basedir,"pickle")
    value = os.path.join(value1,str(name)+".pickle")
    pickle_rec = open(value,"rb")
    recommendation = pickle.load(pickle_rec)
    shuffle(recommendation)
    return render_template('recommendation.html',value=value,recommendation=recommendation[:10])


@app.route('/movie.html/<title>',methods=['GET','POST'])
def movie(title):
    form = MovieRating()
    boolvalue = False
    meanrating = 0

    if form.validate_on_submit():
        rating = int(request.form['slider'])
        comment = form.comment.data
        message = my_tokenizer(comment)
        value = 0
        for word in message:
            try:
                if(word in wordDict.keys()):
                    value = value + wordDict[word]
            except Exception:
                value = value + 0
        
        if(value == 0):
            prob = 0
            meanrating = int(rating)
        else:
            prob =(1/(1 + np.exp(-value)))*5
            meanrating = (prob + int(rating))/2
        
        boolvalue = True
    return render_template('movie.html',form=form,boolvalue=boolvalue,meanrating=meanrating,title=title)

@app.errorhandler(404)
def error_404(error):
    return render_template('error_pages/404.html'),404

@app.errorhandler(403)
def error_403(error):
    return render_template('error_pages/404.html'),403

#run server
if __name__ == '__main__':
    app.run(debug=True)