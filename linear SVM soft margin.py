
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:00:20 2018

@author: LIKS
"""

from sklearn import datasets
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess=tf.Session()
#summary_path
summary_path='./summary'

#import iris data
iris=datasets.load_iris()
x_vals=np.array([[x[0],x[2]] for x in iris.data])
y_vals=np.array([1 if y==0 else -1 for y in iris.target])

#split into train and test data

train_indices=np.random.choice(len(x_vals),round(0.8*len(x_vals)),replace=False)

test_indices=np.array(list(set(range(len(x_vals)))-set(train_indices)))

x_train=x_vals[train_indices]#120*2
y_train=y_vals[train_indices]#[120,]


x_test=x_vals[test_indices]#30*2
y_test=y_vals[test_indices]

#placeholder
x_data=tf.placeholder(shape=[None,2],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)

#model
W=tf.Variable(tf.random_normal(shape=[2,1]),name='W_model')
b=tf.Variable(tf.random_normal(shape=[1,1]),name='b_model')
y_pred=tf.add(tf.matmul(x_data,W),b)#120*1

#loss,L2 norm
#classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
# Put terms together
l2_norm=tf.reduce_sum(tf.square(W))
func_dist_loss=tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y_pred, y_target))))
alpha=tf.constant([0.01])
loss=tf.add(func_dist_loss, tf.multiply(alpha,l2_norm))
print(type(loss),'yes!!!!!!')
tf.summary.scalar('loss_summary',loss)

#minimize loss

y_pred=tf.sign(y_pred)
accuracy=tf.reduce_mean(tf.cast(tf.equal(y_pred,y_target),tf.float32))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
#init,summary
init=tf.global_variables_initializer()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter(summary_path)


#Session
batch_size=100
sess.run(init)

#start training
for i in range(500):
    rand_index=np.random.choice(len(x_train),size=batch_size)
    x_rand=x_train[rand_index]
    y_rand=y_train[rand_index]
    y_rand=np.transpose([y_rand])
    sess.run(train_step,feed_dict={x_data:x_rand,y_target:y_rand})
    loss=sess.run(loss,feed_dict={x_data:x_rand,y_target:y_rand})
    accuracy=sess.run(accuracy,feed_dict={x_data:x_train,y_target:np.transpose([y_train])})
    if (i+1)%10 == 0:
        print('loss=%f,accuracy=%f'%(loss,accuracy))
        #writer.add_summary(merged)
        
    
    
















