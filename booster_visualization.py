from flask import Flask,render_template,request,url_for,redirect,session
from flask_wtf import Form
from wtforms import IntegerField
from flask_cache import Cache  
from wtforms.validators import InputRequired
import random
from booster import *

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
app.config['SECRET_KEY']='Secret'
app.cache = Cache(app)


class tForm(Form):
    '''Simple Integerfield form'''
    n=IntegerField('Iteration Number', default=0,validators=[InputRequired()])

## Initiating booster 
num_points=500 # number of data points
var=.15 #variance of data points (fully separated up to about .1)
n_iter=200 # Number of iterations to compute

@app.cache.cached(timeout=1000, key_prefix="current_time")
def getbooster(n_iter):
    '''Create data and fit num_iter decision tree classifiers to data using boosting method. 
    Data is cashed'''
    booster=Booster()
    booster.moons(num_points,var) #Set X,Y data to be moonlike
    booster.boost(n_iter)
    return booster


@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    booster=getbooster(n_iter) # Get cashed booster object
    n=0
    form=tForm()
    if form.validate_on_submit():
        n=form.n.data # Get n_iter
    # Plot single and combined classification for given n
    return render_template('index.html',image=booster.plot(n,False),
        image2=booster.plot(n,True),form =form,ran=random.random())

if __name__ =='__main__':
    print 'go'
    app.run(debug=True)