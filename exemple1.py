# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)


class TFclass():
  def __init__(self, input_shape, output_shape=[None, 10]):
    self.input = input_shape
    self.output = output_shape
    
  def neural_network(self, x, output_shape):
    x =  tf.reshape(x, [-1, 784])
    nn = tf.layers.dense(x, 300, activation=tf.nn.relu, name='dense1')
    nn2 = tf.layers.dense(nn, 300, activation=tf.nn.relu, name='dense2')
    output = tf.layers.dense(nn2, output_shape, activation=None, name='output')
    return output
  
  def loss(self, logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
  
  def acc(self, logits, y):
    # Compare prediction label and true label
    is_correct = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return accuracy
  
  def main(self):
    
    epochs = 50
    minibatch_size = 250

    steps = int(55000/minibatch_size)
    display_epochs = 100
    x_ = tf.placeholder(tf.float32, self.input)
    y_ = tf.placeholder(tf.float32, self.output)
    train_value = np.zeros(2,)
    model = self.neural_network(x_, self.output[1])
    
    cost = self.loss(logits=model, y=y_)
    accuracy = self.acc(logits=model, y=y_)
    train_step = tf.train.RMSPropOptimizer(0.01).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    # tensorboard
    
    tf.summary.scalar('cross_entropy', cost)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    with tf.Session() as sess:
      
      # Launch the initializer (in a session)
      sess.run(init)
      train_writer = tf.summary.FileWriter('tmp/train', sess.graph)
      test_writer = tf.summary.FileWriter('tmp/test')
      
      for i in range(epochs):

          for j in range(steps):

            # Create minibatch
            batch_X, batch_Y = mnist.train.next_batch(minibatch_size)

            # Define data for the network
            train_data = {x_: batch_X, y_: batch_Y}
            test_data = {x_: mnist.test.images, y_: mnist.test.labels}
            #train_data_full = {x_: mnist.train.images, y_: mnist.train.labels}

            # Training routine of the network
            _, a_train, c_train, summary = sess.run([train_step, accuracy, cost, merged_summary], feed_dict=train_data)
            
            train_value[0] += a_train
            train_value[1] += c_train
            
            # Quality of the model by result of train/test accuracy and cross-entropy value
            if j == steps-1:
                train_acc = 1.0*train_value[0]/steps
                train_loss = 1.0*train_value[1]/steps 
                a_test, c_test, summary_test = sess.run([accuracy, cost, merged_summary], feed_dict=test_data)
                print("Epoch n°" + str(i+1) + " Train accuracy: " + str(train_acc) +
                    " c_e: " + str(train_loss) +
                    " - " + "Test accuracy: " + str(a_test) +
                    " c_e: " + str(c_test))
                
                # write for tensorboard
                summary = tf.Summary()
                summary.value.add(tag="accuracy", simple_value=train_acc)
                summary.value.add(tag="cross_entropy", simple_value=train_loss)
                
                train_writer.add_summary(summary, (i+1))
                train_writer.flush()
                test_writer.add_summary(summary_test, (i+1))
                test_writer.flush()
                
                # reset accuracy
                train_value[0] = 0
                train_value[1] = 0
                
                
                
TF_ = TFclass([None, 28, 28, 1], [None, 10])
TF_.main()
# overfit regarding the cross entropy