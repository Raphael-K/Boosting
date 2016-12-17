from flask import Flask,render_template,request,url_for,redirect,session
from flask_wtf import Form
from wtforms import IntegerField
from flask_cache import Cache  
from wtforms.validators import InputRequired
import mpld3
from helper_functions import *

import random
app = Flask(__name__)

app.config['CACHE_TYPE'] = 'simple'
app.config['SECRET_KEY']='Secret'
app.cache = Cache(app)


def r_template(*args,**kwargs):
    return render_template(*args,**kwargs)



class tForm(Form):
    n=IntegerField('Iteration Number', default=0,validators=[InputRequired()])

@app.cache.cached(timeout=1000, key_prefix="current_time")
def getboosta():
    print 'getboosta'
    X,Y=moons()
    Y=2*Y-1
    boosta=booster(X,Y)
    boosta.boost(100)
    return boosta


@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    boosta=getboosta()
    print boosta.Y[:5]
    n=0
    form=tForm()
    if form.validate_on_submit():
        n=form.n.data
    print n
      
    return r_template('index.html',image=plot_boost(boosta,n,True,False),
        image2=plot_boost(boosta,n,True,True),form =form,ran=random.random())

if __name__ =='__main__':
    print 'go'
    app.run(debug=True)