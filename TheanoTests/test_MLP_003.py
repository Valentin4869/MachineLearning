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


class Layer():
    
    def __init__(self,out_shape=10,in_shape=784, activation=T.tanh):
        self.W=theano.shared(value=n_randGen(0,1,(in_shape,out_shape)),name='W')
        self.b=theano.shared(value=n_randGen(0,1,(out_shape)),name='b')
        self.activation=activation
        self.params=[self.W,self.b]
        

        
    def output(self, X):
       
        return self.activation(T.dot(X,self.W)+self.b)

#        
#        self.o=theano.function(inputs=[X],outputs=[self.output()])        
#    def output2(self,X):
#        return self.o(X)
        
class Seq_model():

    def __init__(self):
        self.layers_N=0
        self.layers=[]
        self.params=[]
        
    def predict(self,X):
        
        return self.net_predict(X)
        
    def add(self, layer):        
        self.layers=self.layers+[layer]
        self.layers_N+=1
        self.params=self.params+layer.params
    
    def create(self, L1_p=0.00, L2_p=0.0001):
        X=T.dmatrix(name='X')
        y=T.ivector(name='y')
        alpha=theano.shared(value=0.01,name='alpha')
        self.net_out=self.layers[0].output(X)
        self.L1=abs(self.layers[0].W).sum()
        self.L2=(self.layers[0].W**2).sum()
        
        for i in range(1,self.layers_N):
            self.net_out=self.layers[i].output(self.net_out)
            self.L1=self.L1+abs(self.layers[i].W).sum()
            self.L2=self.L2+(self.layers[i].W**2).sum()    
                
        self.loss=-T.mean(T.log(self.net_out)[T.arange(0,y.shape[0]),y]) + L1_p*self.L1 + L2_p*self.L2
        g_params=theano.grad(self.loss, self.params)
        pred=T.argmax(self.net_out,axis=1)
        self.net_predict=theano.function(inputs=[X],outputs=pred)
        self.train= theano.function(inputs=[X,y],outputs=[self.loss],updates=[(param, param - alpha*g_param) for param, g_param in zip (self.params, g_params)])
        
        
        
        print('created model')
        
    def fit(self, X, y, epochs=10, batch_size=32):
       
       for i in range(0,epochs):
    
          for bi in range(0,int(np.ceil(D.shape[0]/batch_size))+1):
              print('Epoch '+str(i+1)+'/'+str(epochs)+' -- '
              +str(min((bi+1)*batch_size,samples_N))+'/'+str(X.shape[0])+
              ' -- loss: '+str(self.train(D[bi*batch_size:min((bi+1)*batch_size,X.shape[0])],l[bi*batch_size:min((bi+1)*batch_size,X.shape[0])])[0]))
                                 
            
            
            
 

    
    
    
    
       
       
f = gzip.open('mnist.pkl.gz', 'rb')
(D,l), (_,_), (D_test,l_test) = cPickle.load(f)
l=l.astype('int32')
f.close()
    
#generate random dataset
n_randGen=np.random.normal
mu=0.0
classes=np.asarray([0,1,2,3,4,5,6,7,8,9])
C=classes.shape[0]
(samples_N,m)=D.shape
n_epoch=100
batch_size=32
k=500



l1=Layer(500,784)
l2=Layer(300,500)
l3=Layer(in_shape=500, out_shape=10, activation=T.nnet.softmax) # outlayer
    
    
seq=Seq_model();
seq.add(l1)
seq.add(l3)
print('model layers: ')
print(seq.layers)
seq.create()

seq.fit(D,l,1000,20)
    
print('Performance on test set: '+str(100.0*sum(seq.predict(D_test[0:])==l_test)/l_test.shape[0])+'% accuracy.')
#
#V=theano.shared(value=n_randGen(0,1,(m,k)),name='V')
#c=theano.shared(value=n_randGen(0,1,(k)),name='c')
#W=theano.shared(value=n_randGen(0,1,(k,C)),name='W')
#b=theano.shared(value=n_randGen(0,1,(C)),name='b')
#alpha=theano.shared(value=0.01,name='alpha')
#X=T.dmatrix(name='X')
#y=T.ivector(name='y')
#
#
#P_Y=T.nnet.softmax((T.dot(T.tanh(T.dot(X,V) + c),W)+b))
#predict=T.argmax(P_Y,axis=1)
#log_loss=-T.mean(T.log(P_Y)[T.arange(0,y.shape[0]),y])
#g_W, g_b=T.grad(log_loss,[W,b])
#g_V, g_c=T.grad(log_loss,[V,c])
#
#f_P_Y=theano.function(inputs=[X],outputs=P_Y)
#f_predict=theano.function(inputs=[X],outputs=predict)
#f_train=theano.function(inputs=[X,y],
#                      outputs=[log_loss],
#                      updates=[(W, W-alpha*g_W),(b, b-alpha*g_b), (V, V-alpha*g_V),(c, c-alpha*g_c),(alpha,alpha*1.0)]
#                      )
#
##f_loss=theano.function(inputs=[X,y],outputs=[log_loss])                      
#f_loss=theano.function(inputs=[X,y],outputs=[log_loss])                      
#print('Initial loss: '+str(f_loss(D,l)[0]))
#print('Training for '+ str(n_epoch)+'epochs')
#
#for i in range(0,n_epoch):
#    
#    for bi in range(0,int(np.ceil(D.shape[0]/batch_size))+1):
#        print('Epoch '+str(i+1)+'/'+str(n_epoch)+' -- '
#        +str(min((bi+1)*batch_size,samples_N))+'/'+str(samples_N)+
#        ' -- loss: '+str(f_train(D[bi*batch_size:min((bi+1)*batch_size,samples_N)],l[bi*batch_size:min((bi+1)*batch_size,samples_N)])[0]))
#        
#
#
#h=Layer()
#o=Layer(T.softmax)
#
#
#
#


#print('Performance on test set: '+str(100.0*sum(f_predict(D_test[0:])==l_test)/l_test.shape[0])+'% accuracy.')
#print('probs for ' + str(D[37:38,:]))  
#print(f_P_Y(D[37:38,:]))  
#print(f_predict(D[37:38,:]))  



        
        
        
