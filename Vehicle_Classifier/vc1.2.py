
from utils import *
#^ imports plot, np and stuff

import tensorflow as tf
from tensorflow import Session, zeros, float32, reshape
from tensorflow.examples.tutorials.mnist import input_data



#exec(open("vc1.2.py").read())

#-------------------------------------------------------------------------#
#--------------------------Function Declarations--------------------------#


#evaluate one test sample 
def evin(i,plotit=True):
    print('Accuracy:');
    print(accuracy.eval(session=session,feed_dict={
            X: X_test[i:i+1], y:y_test[i:i+1]}));
    print(session.run(y_out,feed_dict={X:X_test[i:i+1]}));
    print('Actual: ');
    print(y_test[i:i+1])
    if plotit:
        imshow(X_test[i]);

def weight_variable(shape):
   
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


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



#____________________________________
#------------- main() ------------- #


X_train, y_train, X_test, y_test = getCarData();


print('Constructing network\n');
#tf.device('/cpu:0');
save_weights=False;
epochs=10; # 95%-97% accuracy is already attainable early on at around epoch 40. If stuck, restart (unlucky initialization).
train_N=np.shape(X_train)[0];
minibatch_size=128; # small batch size might be better, but larger is faster and 128 is a good compromise
batches=int(np.ceil(train_N/minibatch_size));
in_w=96;
in_h=96;
out_dim=4;

X= tf.placeholder(tf.float32, shape=[None,in_w,in_h,3]);
y= tf.placeholder(tf.float32, shape=[None, out_dim]);

##input_conv1
W_conv1 = weight_variable([5,5, 3, 32]);
b_conv1 = bias_variable([32]);
h_conv1 = tf.nn.relu(conv2d(X, W_conv1)); 
h_pool1 = max_pool_2x2(h_conv1);

##conv1_conv2
W_conv2 = weight_variable([5, 5, 32, 32]);
b_conv2 = bias_variable([32]);
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)); 
h_pool2 = max_pool_2x2(h_conv2);

##conv2_dense1
W_d1 = weight_variable([24 * 24 * 32, 100]);
b_d1 = bias_variable([100]);
h_pool2_flat = tf.reshape(h_pool2, [-1, 24*24*32]);
h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1) + b_d1);

keep_prob_c2d = tf.placeholder(tf.float32)
h_d1_dpt = tf.nn.dropout(h_d1, keep_prob_c2d)

##dense1_dense2
W_d2 = weight_variable([100, 100]);
b_d2 = bias_variable([100]);
h_d2 = tf.nn.relu(tf.matmul(h_d1_dpt, W_d2) + b_d2);

keep_prob_d1d2 = tf.placeholder(tf.float32)
h_d2_dpt = tf.nn.dropout(h_d2, keep_prob_d1d2)


##output
W_out = weight_variable([100, 4]);
b_out = bias_variable([4]);
y_out = tf.matmul(h_d2_dpt, W_out) + b_out; # y_softmax handles scaling, so no activation here should be okay


##loss
y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
ce= tf.reduce_mean(y_softmax);
train_step = tf.train.RMSPropOptimizer(0.001).minimize(ce);

#evaluation
y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));

  
with Session() as session:  
    session.run(tf.global_variables_initializer()); 
    for e in range(0, epochs):
        mean_loss=[];
          
        print("Epoch %i/%i" % (e+1,epochs));
        for batch in range(0, batches):
            
        
            train_accuracy = accuracy.eval(session=session,feed_dict={
                X: X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)], 
                            y: y_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)],keep_prob_c2d:1.0,keep_prob_d1d2:1.0});
            
            mean_loss.append(train_accuracy);
            
           
            if batch%12==0:    
                print('%d/%d | batch accuracy %g | mean epoch accuracy %g' % ((batch+1)*minibatch_size, train_N,
                                                                        train_accuracy, np.mean(mean_loss)));
                

                
            train_step.run(session=session,feed_dict={X: X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)], 
                                y: y_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)],keep_prob_c2d:0.9,keep_prob_d1d2:0.8})
        
              
    
              
        if (e+1)%10==0:
            mean_test_acc=[];
            mean_val_acc=[];

            #evaluating with a loop one at a time because GPU memory is full. Batched should also be okay.
            #0-299: validation
            #300-654: test

            for i in range(0,300):    
                mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
            print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

            for i in range(300,np.shape(y_test)[0]):    
                mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
            print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

            if save_weights:
                tsave('weights/b_conv1.npy',b_conv1,session);
                tsave('weights/b_conv2.npy',b_conv2,session);
                tsave('weights/b_out.npy',b_out,session);
                tsave('weights/b_d2.npy',b_d2,session);
                tsave('weights/b_d1.npy',b_d1,session);
                tsave('weights/W_out.npy',W_out,session);
                tsave('weights/W_d2.npy',W_d2,session);
                tsave('weights/W_d1.npy',W_d1,session);
                tsave('weights/W_conv2.npy',W_conv2,session);
                tsave('weights/W_conv1.npy',W_conv1,session);
        



        
