# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:44:24 2016

"""


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import keras.optimizers as opt
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import scipy.io as sio
from sklearn import cross_validation
import h5py

type_h5=True;
new_model=False
l_path='/opt/lintula/worktmp/001/002/'



if new_model:
    
    if type_h5:
        samples=h5py.File(l_path+'Data/Samples/samples9.mat','r')
        targets=h5py.File(l_path+'Data/Samples/targets9.mat','r')
        samples=samples['samples'];
        targets=targets['targets'];
        samples=np.transpose(samples.value);
        targets=np.transpose(targets.value);
        samples_shape=np.shape(samples);
        targets_shape=np.shape(targets);
    else:
        samples=sio.loadmat('Data/Samples/samples8.mat');
        targets=sio.loadmat('Data/Samples/targets8.mat');
        samples=samples['samples'];
        targets=targets['targets'];
        samples_shape=np.shape(samples);
        targets_shape=np.shape(targets);    
    
    
    
    model2 = Sequential();
    model2.add(LSTM(512, input_shape=(samples_shape[1],samples_shape[2])));
    model2.add(Dense(200));
    

    
    model2.add(Dense((targets_shape[1])));
    model2.add(Activation('linear'));
    
    #optimizer = opt.RMSprop(lr=0.001)
    optimizer = opt.RMSprop(lr=0.001)
    #optimizer=opt.Adam();
    model2.compile(loss='mean_squared_error', optimizer=optimizer)
    
    #X = np.zeros((samples_shape[-1:-4:-1]));
    #X = np.zeros((5,192,300),dtype=float);
    #y = np.zeros((5,192,),dtype=float);
    
    #Shuffle
    
    rs = cross_validation.ShuffleSplit(samples_shape[0], n_iter=1,test_size=0.1, random_state=0)
    
    X_train=[];
    X_test=[];
    y_train=[];
    y_test=[];
    
    for train_index, test_index in rs:
        
        X_train.append(samples[train_index,:,:]);
        y_train.append(targets[train_index,:]);
        X_test.append(samples[test_index,:,:]);
        y_test.append(targets[test_index,:]);
    

model2.fit(X_train[0],y_train[0],nb_epoch=8);

#pred=model2.predict(X_test[0]);

#sio.savemat('preds_002.mat',{'items':pred});
#mode.predict takes a vector of input samples

test_list=X_test[0];
test_shape=np.shape(test_list);


for j in range(0,400):
    pred_N=100; #how many frames to predict
    Predicted_Sequence=np.zeros([np.shape(test_list[0,:,:])[0], pred_N])
    test_ind=j# index of seed, from test samples (X_test)

#seed_seq=np.reshape(test_list[test_ind,:,:],[1,test_shape[1],test_shape[2]])
    seed_seq=np.zeros([1,test_shape[1],test_shape[2]])
#in_seed=test_list[test_ind,:,:];
    seed_seq[0,:,:]=test_list[test_ind,:,:]; 
    in_seed=np.copy(seed_seq[0,:,:]);

    for i in range(0,pred_N):
        pred_i=model2.predict(seed_seq);
        Predicted_Sequence[:,i]=pred_i
        seed_seq[0,:,:]=np.concatenate((seed_seq[0,:,1:],np.transpose(pred_i)),axis=1);

        sio.savemat('Data/Samples/rn_test/seedlist_'+ str(j)+'.mat',{'items':in_seed});     
        sio.savemat('Data/Samples/rn_test/predlist_'+ str(j)+'.mat',{'items':Predicted_Sequence});    

#sio.savemat('Data/Samples/test_indices.mat',{'items':test_index});     