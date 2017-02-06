# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:58:57 2017

"""
first_time=False

if first_time:
    import theano
    import theano.tensor as T
    import numpy as np

    
#generate random dataset
n_randGen=np.random.normal
mu=0.0
classes=np.asarray([0.2,1.5,3.4,7]) #sigma values
C=4
m=3
samples_N=1000
n_epoch=64

#4000x3 data set of x in R^{3} from 4 classes

D=np.zeros((C*samples_N,m))
l=np.zeros((C*samples_N),dtype='int32')

for i in range(C):
  D[i*1000:(i+1)*1000,:]=n_randGen(mu,classes[i],(1000,m))
  l[i*1000:(i+1)*1000]=i;
  
# W: weights matrix; column i ---> parameters of class_i; R^{3}
# b: bias column vector   : element i --> free parameter for class_i; R^{1}

W=theano.shared(value=n_randGen(0,1,(m,C)),name='W')
b=theano.shared(value=n_randGen(0,1,(C)),name='b')
alpha=theano.shared(value=0.08,name='alpha')
X=T.dmatrix(name='X')
y=T.ivector(name='y')


P_Y=T.nnet.softmax((T.dot(X,W)+b))
predict=T.argmax(P_Y)
log_loss=-T.mean(T.log(P_Y)[T.arange(0,y.shape[0]),y])
g_W, g_b=T.grad(log_loss,[W,b])

f_P_Y=theano.function(inputs=[X],outputs=P_Y)
f_predict=theano.function(inputs=[X],outputs=predict)

train=theano.function(inputs=[X,y],
                      outputs=[log_loss],
                      updates=[(W, W-alpha*g_W),(b, b-alpha*g_b)]
                      )

#f_loss=theano.function(inputs=[X,y],outputs=[log_loss])                      
f_loss=theano.function(inputs=[X,y],outputs=[log_loss])                      
print('Initial loss: '+str(f_loss(D,l)[0]))
print('Training for '+ str(n_epoch)+'epochs')

for i in range(0,n_epoch):
    print('Epoch '+str(i)+'/'+str(n_epoch)+'\nloss: '+str(train(D,l)[0]))


#print('probs for ' + str(D[37:38,:]))  
#print(f_P_Y(D[37:38,:]))  
#print(f_predict(D[37:38,:]))  