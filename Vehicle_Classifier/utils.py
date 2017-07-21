import os
import glob
import socket
import numpy as np
from numpy import reshape, shape
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

def dims(x):
    return shape(shape(x))[0];

def tsave(fname,tensor_x,session):
    host_x= session.run(tensor_x);
    np.save(fname,host_x);

#matlab style
def conv2(x,f): 
    return tf.nn.conv2d(x,f, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):

  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

def run(fn):
    exec(open(fn+".py").read())

def imsave(res,fname='c.csv'):
    if dims(res)<3:
        np.savetxt(fname, res, delimiter=",") ;
    elif dims(res)==3:
        print('Warning: saving only first channel');
        np.savetxt(fname, res[:,:,0], delimiter=",");
    elif dims(res)==4:
        print('Warning: saving only first feature map (4D input)');
        np.savetxt(fname, res[0,:,:,0], delimiter=",");

    else:
        print('can\'t save this...');

def imshow(im):
    matplotlib.pyplot.imshow(im);
    matplotlib.pyplot.show();

def imread(fname):
    im=plt.imread(fname)
    im = (im - 0.0) /255.0 ;#im2double
    return im;

def imread2(fname):
    im=plt.imread(fname)
    im = (im - 0.0) /255.0 ;#im2double
    return reshape(im,[1,shape(im)[0],shape(im)[1],shape(im)[2]]);

def getClassStr(class_i):
    if class_i==0:
        return 'BUS';
    elif class_i==1:
        return 'NORMAL';
    elif class_i==2:
        return 'TRUCK';
    elif class_i==3:
        return 'VAN';

def getCarData():
    path_train='E:/vc/DATA/TRAIN/';
    path_test='E:/vc/DATA/TEST/';        


    print('Collecting Data\n');
    train_fnamelist=os.listdir(path_train);
    test_fnamelist=os.listdir(path_test);

    BUS_fnamelist_train=glob.glob(path_train+'BUS/*.jpg');
    NORMAL_fnamelist_train=glob.glob(path_train+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_train=glob.glob(path_train+'TRUCK/*.jpg');
    VAN_fnamelist_train=glob.glob(path_train+'VAN/*.jpg');

    BUS_fnamelist_test=glob.glob(path_test+'BUS/*.jpg');
    NORMAL_fnamelist_test=glob.glob(path_test+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_test=glob.glob(path_test+'TRUCK/*.jpg');
    VAN_fnamelist_test=glob.glob(path_test+'VAN/*.jpg');

    X_train = [];
    y_train= np.zeros([len(BUS_fnamelist_train)+
    len(NORMAL_fnamelist_train)+len(TRUCK_fnamelist_train)+len(VAN_fnamelist_train),4])

    X_test=[];
    y_test= np.zeros([len(BUS_fnamelist_test)+
    len(NORMAL_fnamelist_test)+len(TRUCK_fnamelist_test)+len(VAN_fnamelist_test),4])
    
    offset=0;
  
 
    for i in range(offset,len(BUS_fnamelist_train)):
        X_train.append(plt.imread(BUS_fnamelist_train[i]));
        y_train[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_train)):
        X_train.append(plt.imread(NORMAL_fnamelist_train[i]));
        y_train[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_train)):
        X_train.append(plt.imread(TRUCK_fnamelist_train[i]));
        y_train[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_train)):
        X_train.append(plt.imread(VAN_fnamelist_train[i]));
        y_train[i+offset][3]=1;


    offset=0;
    
    for i in range(offset,len(BUS_fnamelist_test)):
        X_test.append(plt.imread(BUS_fnamelist_test[i]));
        y_test[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_test)):
        X_test.append(plt.imread(NORMAL_fnamelist_test[i]));
        y_test[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_test)):
        X_test.append(plt.imread(TRUCK_fnamelist_test[i]));
        y_test[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_test)):
        X_test.append(plt.imread(VAN_fnamelist_test[i]));
        y_test[i+offset][3]=1;


    print('Shuffling...')
    #Shuffle
    rndIdx=np.random.permutation((np.shape(X_train))[0]); 
    cX_train=np.copy(X_train);
    cy_train=np.copy(y_train);

    for i in range(0,np.shape(X_train)[0]):
        X_train[i]=cX_train[rndIdx[i]];
        y_train[i]=cy_train[rndIdx[i]];

    rndIdx=np.random.permutation((np.shape(X_test))[0]); 
    cX_test=np.copy(X_test);
    cy_test=np.copy(y_test);
    
    for i in range(0,np.shape(X_test)[0]):
        X_test[i]=cX_test[rndIdx[i]];
        y_test[i]=cy_test[rndIdx[i]];
    
    return X_train, y_train, X_test, y_test