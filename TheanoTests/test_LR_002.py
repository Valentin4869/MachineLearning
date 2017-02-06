# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:58:57 2017

"""
first_time=True

if first_time:
    import theano
    import theano.tensor as T
    import numpy as np

    
#generate random dataset
n_randGen=np.random.normal
mu=0.0
classes=np.asarray([0.2,17.5,100]) #sigma values
C=classes.shape[0]
m=3
samples_C=10000
samples_N=samples_C*C
n_epoch=64
batch_size=512
#4000x3 data set of x in R^{3} from 4 classes

D=np.zeros((C*samples_C,m))
l=np.zeros((C*samples_C),dtype='int32')
l[:]=np.random.randint(0,C,np.shape(l)) # random classes

for i in range(0,C):
  D[i*samples_C:(i+1)*samples_C,:]=n_randGen(mu,classes[l[i]],(samples_C,m))
  
  
#Shuffle

  
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
f_train=theano.function(inputs=[X,y],
                      outputs=[log_loss],
                      updates=[(W, W-alpha*g_W),(b, b-alpha*g_b), (alpha,alpha*0.95)]
                      )

#f_loss=theano.function(inputs=[X,y],outputs=[log_loss])                      
f_loss=theano.function(inputs=[X,y],outputs=[log_loss])                      
print('Initial loss: '+str(f_loss(D,l)[0]))
print('Training for '+ str(n_epoch)+'epochs')

for i in range(0,n_epoch):
    print('Epoch '+str(i+1)+'/'+str(n_epoch))
    
    for bi in range(0,int(np.ceil(D.shape[0]/batch_size))):
        print('loss: '+str(f_train(D[bi*batch_size:min((bi+1)*batch_size,samples_N)],l[bi*batch_size:min((bi+1)*batch_size,samples_N)])[0]))
        


rn_class=np.random.randint(0,C)
print('Prediction for a class '+ str(rn_class) +' sample: '+str(f_predict(n_randGen(mu,classes[rn_class],(1,m)))))
#print('probs for ' + str(D[37:38,:]))  
#print(f_P_Y(D[37:38,:]))  
#print(f_predict(D[37:38,:]))  