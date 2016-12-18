from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import mpld3
import time


def plot_boost(boosta,n,classifiers=True,combined=True):
    t0=time.time()

    X=boosta.X
    Y=boosta.Y
    c1='red'
    c0='blue'

    cmap = colors.ListedColormap([c0, c1])   
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(5,5))
    fig.set_frameon(False) 
    ax = fig.add_subplot(111)
    x_min,x_max = X[:,0].min()*1.1 , X[:,0].max()*1.1
    y_min,y_max = X[:,1].min()*1.1 , X[:,1].max()*1.1 

    if n!=0:

        xx, yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
        if combined== True:
            Z = boosta.predict(np.c_[xx.ravel(), yy.ravel()],n)
        else:
            Z = boosta.classifiers[n-1].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z,cmap=cmap,norm=norm,alpha=.3)
        #ax.imshow(Z.T,extent=[x_min,x_max,y_min,y_max],cmap=cmap,norm=norm,origin='lower',alpha=.3)
    else:
        n=1

    idx_1=np.where(Y==1)
    idx_0=np.where(Y==-1)
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])

    ax.scatter(X[idx_1,0],X[idx_1,1],marker='o',c=c1,label='1',s=6+50*boosta.weights[n-1][idx_1])
    ax.scatter(X[idx_0,0],X[idx_0,1],c=c0,label='-1',s=6+50*boosta.weights[n-1][idx_0])
    plt.axis('off')
    t1=time.time()
    result=mpld3.fig_to_html(fig)
    t2=time.time()
    print "Computation: {0} s".format(t1-t0)
    print "Conversion: {0} s".format(t2-t1)
    return result




    
class Booster:
    ''' Booster class for given base-classifier(TODO) and on given X,Y data'''
    
    def __init__(self):
        self.X=np.random.rand(100,2)
        self.Y=(np.random.randint(2,100)*2)-1

    def moons(self,num,var):
        '''make moon-shaped data'''
        self.X,self.Y=make_moons(num,noise=var,random_state=1)
        self.Y=2*self.Y-1

        
    def boost(self,n_iter):  
        '''Calculate classifiers, weights and errors for a given iteration number'''
        self.classifiers=[]
        self.alphas=np.zeros(n_iter)
        self.weights=np.ones((n_iter+1,self.Y.size))/self.Y.size
        
        for i in range(n_iter):            
            #Classify
            clf=DecisionTreeClassifier(max_depth=1)
            clf.fit(self.X,self.Y,sample_weight=self.weights[i])
            self.classifiers.append(clf)
            
            #Predict
            pred=clf.predict(self.X)

            #Update errors
            err=1-clf.score(self.X,self.Y,sample_weight=self.weights[i])            
            alpha=0.5*np.log((1-err)/err)
            self.alphas[i]=alpha
            
            #Update Weights
            self.weights[i+1]=normalize(self.weights[i]*np.exp(-alpha*self.Y*pred))

 
    def predict(self,X,n):
        ''' Predict labels of data set X using first n classifiers'''

        #Create prediction Matrix
        pred_matrix=np.zeros((n,X.shape[0]))
        for i in range(n):
            pred_matrix[i]=self.classifiers[i].predict(X)
            
        #Final prediction
        return np.sign(np.dot(pred_matrix.T,self.alphas[:n]))

    def plot(self,n,combined=True):
        ''' Plot data of iteration n. Decision boundary is shown for all classifiers 
        up to n (combined=True) or only for classifier n (False) '''
        t0=time.time()
        X=self.X
        Y=self.Y
        #Prepare colors for scatter and decision boundary contourplot
        c1='red'
        c0='blue'
        cmap = colors.ListedColormap([c0, c1])   
        bounds=[-2,0,2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        n_steps=50
        #Plot initiation
        fig = plt.figure(figsize=(5,5))
        fig.set_frameon(False) 
        ax = fig.add_subplot(111)
        x_min,x_max = X[:,0].min()*1.1 , X[:,0].max()*1.1
        y_min,y_max = X[:,1].min()*1.1 , X[:,1].max()*1.1 
        ax.set_xlim([x_min,x_max])
        ax.set_ylim([y_min,y_max])
        plt.axis('off')

        if n!=0: #Plot the decision boundary if not initial state

            xx, yy = np.meshgrid(np.linspace(x_min,x_max,n_steps),np.linspace(y_min,y_max,n_steps))
            if combined== True:
                Z = self.predict(np.c_[xx.ravel(), yy.ravel()],n)
            else:
                Z = self.classifiers[n-1].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = ax.contourf(xx, yy, Z,cmap=cmap,norm=norm,alpha=.3)
        else:
            n=1

        # Scatter plot showing label and weights of data
        idx_1=np.where(Y==1)
        idx_0=np.where(Y==-1)
        ax.scatter(X[idx_1,0],X[idx_1,1],marker='o',c=c1,label='1',s=6+50*self.weights[n-1][idx_1])
        ax.scatter(X[idx_0,0],X[idx_0,1],c=c0,label='-1',s=6+50*self.weights[n-1][idx_0])

        #Timing and conversion
        t1=time.time()
        result=mpld3.fig_to_html(fig)
        t2=time.time()
        print "Computation: {0} s".format(t1-t0)
        print "Conversion: {0} s".format(t2-t1)
        return result
        
