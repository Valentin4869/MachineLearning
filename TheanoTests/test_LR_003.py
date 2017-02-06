# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:58:57 2017

"""
first_time=True

if first_time:
    import theano
    import theano.tensor as T
    import numpy as np
    import cPickle, gzip



f = gzip.open('mnist.pkl.gz', 'rb')
(D,l), (_,_), (D_test,l_test) = cPickle.load(f)
l=l.astype('int32')
f.close()
    
#generate random dataset
n_randGen=np.random.normal
mu=0.0
classes=np.asarray([0,1,2,3,4,5,6,7,8,9]) #sigma values
C=classes.shape[0]
(samples_N,m)=D.shape
n_epoch=32
batch_size=64
#4000x3 data set of x in R^{3} from 4 classes



  
# W: weights matrix; column i ---> parameters of class_i; R^{3}
# b: bias column vector   : element i --> free parameter for class_i; R^{1}

W=theano.shared(value=n_randGen(0,1,(m,C)),name='W')
b=theano.shared(value=n_randGen(0,1,(C)),name='b')
alpha=theano.shared(value=0.09,name='alpha')
X=T.dmatrix(name='X')
y=T.ivector(name='y')


P_Y=T.nnet.softmax((T.dot(X,W)+b))
predict=T.argmax(P_Y,axis=1)
log_loss=-T.mean(T.log(P_Y)[T.arange(0,y.shape[0]),y])
g_W, g_b=T.grad(log_loss,[W,b])

f_P_Y=theano.function(inputs=[X],outputs=P_Y)
f_predict=theano.function(inputs=[X],outputs=predict)
f_train=theano.function(inputs=[X,y],
                      outputs=[log_loss],
                      updates=[(W, W-alpha*g_W),(b, b-alpha*g_b), (alpha,alpha)]
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
print('Performance on test set: '+str(100.0*sum(f_predict(D_test[0:])==l_test)/l_test.shape[0])+'% accuracy.')
#print('probs for ' + str(D[37:38,:]))  
#print(f_P_Y(D[37:38,:]))  
#print(f_predict(D[37:38,:]))  