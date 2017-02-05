# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:53:49 2016

"""

import numpy as np
import csv
import  help_functions as hf
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import SVC

path='D:/0/copper/'
blocks=1
skip=0

if (1-skip):
    train_flist, train_n=hf.getcsvdata() #get list of file names from csv
    train_n=train_n

hbins=np.arange(0,257,1)
hbins[-1]=256


print "# training files found:",
print train_n
 
X,y=hf.getblocklbp(train_flist, train_n, blocks=blocks, bins=hbins)

#save for reference:
y_log=y
X_log=X

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)


model5=SVC(kernel='poly', probability=True)
model5.fit(X, y)

cvs=cross_val_score(model5, X, y, cv=10)
print 'SVM: ',
#print score     
print "cvs: ", cvs.mean(),    
print cvs    

#----------test------------
test_blocks=1  
test_flist, test_n=hf.getcsvdata(path+'/test.csv')
test_csv=open(path+'/testw.csv', 'wb')
writer=csv.writer(test_csv)
tX=[]
tX.append(['Id', 'Prediction'])

X,y=hf.getblocklbp(test_flist, test_n, blocks=test_blocks, bins=hbins, path='d:/0/copper/test/', test=1)

y_pred=model5.predict(X)


for i in range(0,test_n):   
    tX.append([(test_flist[i])[0], str(int(y_pred[i]))])

writer.writerows(tX)
test_csv.close()
