# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 14:59:50 2016

@author: Hex
"""
import os
import csv;
import numpy as np;
from matplotlib import pyplot as plt;
from math import ceil, floor;
from skimage.feature import local_binary_pattern
import scipy;
from numpy.random import randint as randi;
from sklearn.preprocessing import scale;
from skimage import exposure;
#getlist data from dsv in train_flist and its size in rows
def getcsvdata(path0='d:/0/copper/train.csv', ignorefirst=1):
    
    train_csv=open(path0);
    reader=csv.reader(train_csv);
    train_flist=[];
    train_n= 0;
    
    if ignorefirst:
        reader.next();
    
    for row in reader:
        train_flist.append(row);
        train_n+=1;
    
    return train_flist, train_n
    
    
def getflist(path='d:/0/copper/train_g/'):
    flist=os.listdir(path);
    return flist, int(np.shape(flist)[0]);    

    
    
#create list of labeled features as histograms from a list of files
def gethist(flist, train_n, bins=[0, 32, 64, 128, 256], path='d:/0/copper/train_g2/', test=0):
    m=len(bins);
    if test:
        X=np.zeros((train_n,m-1));
        y=np.zeros((train_n,   ));
        
    else:
        X=np.zeros((train_n*4,m-1));
        y=np.zeros((train_n*4,   ));
        
   # bins=np.ones((1,bins_n))*bins;
    if test:
        for i in range(0,train_n):
            im= plt.imread(path + flist[i][0]);        
            
            #im=np.mean(im,-1)
            histo=np.histogram(np.asarray(im).reshape(-1),bins);
            
            X[i][0:]=histo[0];   
            
        
    else:        
        for i in range(0,train_n):
            im= plt.imread(path + flist[i][0]);        
            
            imh= np.fliplr(im);
            imv= np.flipud(im);
            imhv=np.fliplr(imv);
            im=np.mean(im,-1)
            histo=np.histogram(np.asarray(im).reshape(-1),bins);
            
            X[i*4+0][0:]=histo[0];   
            if (1-test):    
                y[i*4+0]=(flist[i][1]);
            
            histo=np.histogram(np.asarray(imh).reshape(-1),bins);
            
            X[i*4+1][0:]=histo[0];   
            if (1-test):    
                y[i*4+1]=(flist[i][1]);
            
            histo=np.histogram(np.asarray(imv).reshape(-1),bins);
            
            X[i*4+2][0:]=histo[0];   
            if (1-test):    
                y[i*4+2]=(flist[i][1]);
            
            histo=np.histogram(np.asarray(imhv).reshape(-1),bins);
            
            X[i*4+3][0:]=histo[0];   
            if (1-test):    
                y[i*4+3]=(flist[i][1]);
            
        
    return X, y;
    
#train: fill list with smaller crops/blocks of each training image, then use each individually
#test: get one block from each test image, but generate several sets where the chosen block 
#is different for each set for each test image. Average results from each classifier.    
def getblockhist(flist, train_n, bins=[0, 32, 64, 128, 256], blocks=3, blockid=0,path='d:/0/copper/train_g/', test=0):
    m=len(bins);
    if test:
        X=np.zeros((train_n,m-1));
        y=np.zeros((train_n,   ));
        
    else:
        X=np.zeros((train_n*blocks,m-1));
        y=np.zeros((train_n*blocks,   ));
        
   # bins=np.ones((1,bins_n))*bins;
    if test:
        for i in range(0,train_n):
            im= randblock(plt.imread(path + flist[i][0]),128);        
            
            #im=np.mean(im,-1)
            histo=np.histogram(np.asarray(im).reshape(-1),bins);
            
            X[i][0:]=histo[0];   
            
        
    else:        
        for i in range(0,train_n):
            im= plt.imread(path + flist[i][0]);        
            
            for j in range(0, blocks):
                
                histo=np.histogram(np.asarray(randblock(im,128)).reshape(-1),bins);
                
                X[i*blocks + j][0:]=histo[0];   
                if (1-test):   # if data is not for testing  
                    y[i*blocks + j]=(flist[i][1]);
                
            
        
    return X, y;
    
def getlbp(flist, train_n, bins=[0, 32, 128, 160,  256], path='d:/0/copper/train_g/'):
   
    m=len(bins);   
   
    X=np.zeros((train_n,m-1));
    y=np.zeros((train_n,   ));
    radius = 6;
    n_points = 8 * radius;
    METHOD = 'uniform';

    
#load all files from C1 and C2, put in X with their corresponding labels in y
    for i in range(0,train_n):
        radius = randi(3,9,1);
        n_points = randi(4,20,1) * radius;
        im= plt.imread(path + flist[i][0]);
       
        im_lbp= local_binary_pattern(im, n_points, radius, METHOD);
        histo=np.histogram(np.asarray(im_lbp).reshape(-1),bins=bins, range=(0.0, 1.0));
        X[i][0:]=histo[0];
        y[i]=(flist[i][1]);
        
    return X, y;
    
    
def getblocklbp(flist, train_n, blocks=3, bins=[0, 32, 128, 160,  256], path='d:/0/copper/train/', test=0):
    print('Extracting LBP for each block')
    m=len(bins);   
    X=np.zeros((train_n*blocks,m-1));
    
    y=np.zeros((train_n*blocks,   ));   
    n_points = 8;
    radius =3;
    METHOD = 'ror';
    print "Set_size: ",;
    print train_n*blocks, '\n';
    
    track_i=0;
    st=train_n/8;
#load all files from C1 and C2, put in X with their corresponding labels in y
    for i in range(0,train_n):
        
        im= (plt.imread(path + flist[i][0]));
        
        if i>=st*track_i:
            print 100*i/train_n,'%\n',
            track_i+=1;
        
        p2, p98 = np.percentile(im, (2, 98))
#        im_block = exposure.rescale_intensity(im, in_range=(p2, p98))
#        im_block = exposure.equalize_hist(im_block);
       # im[:,:,2]=im[:,:,2]+6
      #  im_block=exposure.adjust_sigmoid(im,0.38,16.8)
      #  im_block=exposure.adjust_sigmoid(im,0.34,19)
        #im_block=exposure.adjust_sigmoid(im,0.35,19) 86.8
        #im_block=exposure.adjust_sigmoid(im,0.38,15.5) 86.85
        im_block=exposure.adjust_sigmoid(im,0.38,16.45)
        im_block = exposure.equalize_hist(im_block);            
        im_lbp= local_binary_pattern(im_block[:,:,2], n_points, radius);
        #im_lbp=whsblock(im_lbp1,0,0,509);   
          
        
        for j in range(0,blocks):
            
            #im_lbp= randblock(im_lbp1,randi(400,509,1));
           #im_block=randblock(im,randi(490,512,1));
            
#            im_base=im_lbp;
#                
#            im_base= randblock(im_lbp,randi(475,505,1))
#            if i%100==0:
#                scipy.misc.imsave('d:/0/copper/blocks/'+ flist[i][0]+'_'+str(j)+'.jpg',im_block);
#               # scipy.misc.imsave('d:/0/copper/blocks/'+ flist[i][0]+'_'+str(j)+'_filter.jpg',im_block[:,:,0]);
#                scipy.misc.imsave('d:/0/copper/blocks/'+ flist[i][0]+str(j)+str(j)+'.jpg',im_lbp1);
#                scipy.misc.imsave('d:/0/copper/blocks/'+ flist[i][0]+str(j)+str(j)+'_lbp2.jpg',im_lbp);
#                
            histo=np.histogram(np.asarray(im_lbp).reshape(-1),  bins=bins);
            
            X[i*blocks +j][0:]=histo[0];
            if (1-test):
                y[i*blocks+j]=(flist[i][1]);
                
                
            X[i*blocks +j]=scale(X[i*blocks +j]);
#            
#            min_i=np.min(X[i*blocks +j]);
#            max_i=np.max(X[i*blocks +j]);
#            X[i*blocks +j]=np.subtract(X[i*blocks +j], min_i);
#            X[i*blocks +j]=np.divide(X[i*blocks +j], max_i);
            
        
    return X, y;
        
    
def getvariance(flist, train_n,  path='d:/0/copper/train/'):
    
    X=np.zeros((train_n,1));
    y=np.zeros((train_n,   ));

   # bins=np.ones((1,bins_n))*bins;
    
    for i in range(0,train_n):
        im= plt.imread(path + flist[i][0]);
        
        X[i][0:]=np.var(im[0:][0:][0:]);        
        y[i]=(flist[i][1]);
        
    return X, y;
    
def bilinear(image, r, c):
    minr = floor(r)
    minc = floor(c)
    maxr = ceil(r)
    maxc = ceil(c)
    
    dr = r-minr
    dc = c-minc
    
    top = (1-dc)*image[minr,minc] + dc*image[minr,maxc]
    bot = (1-dc)*image[maxr,minc] + dc*image[maxr,maxc]

    return (1-dr)*top+dr*bot

def lbp1(image, P=8, R=1):
    rr = - R * np.sin(2*np.pi*np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2*np.pi*np.arange(P, dtype=np.double) / P)
    rp = np.round(rr, 5)
    cp = np.round(cc, 5)
    
    rows = image.shape[0]
    cols = image.shape[1]

    output = np.zeros((rows, cols))

    for r in range(R,rows-R):
        for c in range(R,cols-R):
            lbp = 0
            for i in range(P):
                if bilinear(image, r+rp[i], c+cp[i]) - image[r,c] >= 0:
                    lbp += 1<<i
                            
            output[r,c] = lbp

    return output

def makegrayscale(source_path, target_path):
    

    flist, n= getflist(source_path);

    for i in range(0,n):
        im=plt.imread(source_path + flist[i]);
        im=np.mean(im,-1);
        scipy.misc.imsave(target_path+flist[i],im);
    
def whsblock(im,h,w,s=50):
    dims=np.shape(im)
    r1=min(h,dims[0]-s);
    r2=min(r1+s,dims[0]);
    r3=min(w,dims[1]-s);
    r4=min(r3+s,dims[1]);  
   
    return im[r1:r2,r3:r4];


def randblock(im,s=50):
    dims=np.shape(im)
    
    return whsblock(im,randi(0,dims[0]-s+1,1),randi(0,dims[1]-s+1,1), s);
    
#test only. Average the predictions over all blocks    
def getavgprob(blockprob, blocks,test_n):
    m_prob=np.zeros((test_n,6));
    
    for i in range(0,test_n):
        m_prob[i]=np.mean(blockprob[i*blocks:i*blocks+blocks],0);
        
    return m_prob;

def getpredprob(prob, classes):
    fsize=np.shape(prob);
    fpred=[];
    for i in range(0,fsize[0]):
        fpred.append(classes[np.argmax(prob[i])]); 
        
    return fpred;    
    