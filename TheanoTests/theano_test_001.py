# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:58:57 2017

"""

reimport=True

if reimport:
    import theano
    import theano.tensor as T
    import numpy as np

    reimport=False
    
#generate random dataset
n_randGen=np.random.normal
mu=0.0
classes=np.asarray([0.2,1.5,3.4,7]) #sigma values
C=4
samples_N=1000
m=3

#4000x3 data set of x in R^{3} from 4 classes

D=np.zeros((C*samples_N,m));

for i in range(C):
  D[i*1000:(i+1)*1000,:]=n_randGen(mu,classes[i],(1000,m))
  
  
# W: weights matrix; column i ---> parameters of class_i; R^{3}
# b: bias column vector   : element i --> free parameter for class_i; R^{1}

W=theano.shared(value=n_randGen(0,1,(m,C)),name='W')
b=theano.shared(value=n_randGen(0,1,(C)),name='b')


X=T.dmatrix(name='X')
Y=T.dmatrix(name='Y')

P_Y=T.nnet.softmax((T.dot(X,W)+b))
predict=T.argmax(P_Y)

f_P_Y=theano.function(inputs=[X],outputs=P_Y)
f_predict=theano.function(inputs=[X],outputs=predict)


print('probs for ' + str(D[37:38,:]))  
print(f_P_Y(D[37:38,:]))  
print(f_predict(D[37:38,:]))  