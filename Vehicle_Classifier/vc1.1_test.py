
from utils import *
#^ imports plot, np and stuff

import tensorflow as tf
from tensorflow import Session, zeros, float32, reshape
from tensorflow.examples.tutorials.mnist import input_data

def imshow(im):
    matplotlib.pyplot.imshow(im);
    matplotlib.pyplot.show();



#exec(open("vc1.1_test.py").read())

#-------------------------------------------------------------------------#
#--------------------------Function Declarations--------------------------#

def evin(i,plotit=True):
    print('___________________________')
    print('y_out:');
    print(session.run(y_out,feed_dict={X:X_test[i:i+1],
            keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
    print('y_softmax:');
    print(session.run(tf.nn.softmax(y_out),feed_dict={X:X_test[i:i+1],
            keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
    print('y_softmax CE:');
    print(1.0-session.run(y_softmax,feed_dict={X:X_test[i:i+1],y: y_test[i:i+1],
            keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
    print('\nPredicted Class: ' + getClassStr(np.argmax(session.run(y_out,feed_dict={X:X_test[i:i+1],y: y_test[i:i+1], keep_prob_c2d:1.0,keep_prob_d1d2:1.0}))))
    print('True Class: ' + getClassStr(np.argmax(y_test[i:i+1])));

    if plotit:
        imshow(X_test[i]);

def ev_mistakes(plotit=True):
    for i in range(0,test_N):
        if np.argmax(session.run(y_out,feed_dict={X:X_test[i:i+1],y: y_test[i:i+1], keep_prob_c2d:1.0,keep_prob_d1d2:1.0})) != np.argmax(y_test[i:i+1]):
            evin(i,plotit);


def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.01)  
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):

  return tf.nn.conv2d(x,
                      W,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

def max_pool_2x2(x):

  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')


#-------------------------------------------------------------------------#


#____________________________________
#------------- main() ------------- #
#____________________________________

#load saved weights and manually evaluate network
first_time=True;

if first_time:
    X_train, y_train, X_test, y_test = getCarData();
    first_time=False;

print('Constructing network\n');

train_N=np.shape(X_train)[0];
test_N=np.shape(X_test)[0];
minibatch_size=128;
batches=int(np.ceil(train_N/minibatch_size));

in_w=96;
in_h=96;
out_dim=4;

X= tf.placeholder(tf.float32, shape=[None,in_w,in_h,3]);

y= tf.placeholder(tf.float32, shape=[None, out_dim]);



##input_conv1


W_conv1 = tf.Variable(np.load('weights/acc_977_972/W_conv1.npy'));
h_conv1 = tf.nn.relu(conv2d(X, W_conv1)); 
h_pool1 = max_pool_2x2(h_conv1);


##conv1_conv2
W_conv2 = tf.Variable(np.load('weights/acc_977_972/W_conv2.npy'));
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)); 
h_pool2 = max_pool_2x2(h_conv2);


##conv2_dense1
W_d1 = tf.Variable(np.load('weights/acc_977_972/W_d1.npy'));
b_d1 = tf.Variable(np.load('weights/acc_977_972/b_d1.npy'));
h_pool2_flat = tf.reshape(h_pool2, [-1, 24*24*32]);
h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1) + b_d1);


keep_prob_c2d = tf.placeholder(tf.float32)
h_d1_dpt = tf.nn.dropout(h_d1, keep_prob_c2d)

##dense1_dense2
W_d2 = tf.Variable(np.load('weights/acc_977_972/W_d2.npy'));
b_d2 = tf.Variable(np.load('weights/acc_977_972/b_d2.npy'));
h_d2 = tf.nn.relu(tf.matmul(h_d1_dpt, W_d2) + b_d2);

keep_prob_d1d2 = tf.placeholder(tf.float32)
h_d2_dpt = tf.nn.dropout(h_d2, keep_prob_d1d2)


##output
W_out = tf.Variable(np.load('weights/acc_977_972/W_out.npy'));
b_out = tf.Variable(np.load('weights/acc_977_972/b_out.npy'));
y_out = tf.matmul(h_d2_dpt, W_out) + b_out; # y_softmax handles scaling, so this should be okay


##loss
y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
ce= tf.reduce_mean(y_softmax);
train_step = tf.train.RMSPropOptimizer(0.001).minimize(ce);

#evaluation
y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));



session=Session(); 
session.run(tf.global_variables_initializer()); 
    
mean_test_acc=[];
        
for i in range(0,np.shape(y_test)[0]):    
    mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));

print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

#manual evaluation; comment out if not needed
#for i in range(0,test_N):
#    evin(i);

ev_mistakes()
    
session.close()


        
