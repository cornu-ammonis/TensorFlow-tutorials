#constructed via the following tutorial: https://www.tensorflow.org/get_started/mnist/beginners
# May contain modifications and deviations from original source - ACJ 2/16/2017
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main(_):
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #this is a placeholder -- one can input any number of MNIST images flattened into a 784d vector
    #here None means that a dimension can be of any length

    x = tf.placeholder(tf.float32, [None, 784])
    
    #initializes Weights and biases as tensors full of 0s
    #note W is [784, 10] because we want to multiply the 784d vectors by it to produce 10d vectors of evidence for
    #each class (per image)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    #to get output of batches x 10weights,
    #one must do tf.matmul(x, W) because x is batchsize x numberpixels and W is numPixels x 10

    #defines model
    y = tf.matmul(x, W) + b

    #placeholder to input correct answers
    y_ = tf.placeholder(tf.float32, [None, 10])

    #--first convolutional layer--

    #will compute 32 features for each 5x5 patch. first 2 dimensions are patch size, next is number input channels
    #last is the number of output channels
    W_conv1 = weight_variable([5, 5, 1, 32])

    #bias vector with component for each output channel
    b_conv1 = bias_variable([32])


    #reshapes x as a 4d tensor, 2nd and 3rd are width and height, 4th is number color channels
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #convolve x_image with weight tensor, add bias, apply ReLU, and then max pool. max pool method will reduce
    #image to 14x14
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    #uses built in tf.nn.softmax_cross_entropy_with_logits because raw formulation of cross entropy can be unstable
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    #tells tensorflow to minimize cross_entropy by using gradient descent with rate .5 - this shifts each
    #variable a bit in the direction which minimizes cross entropy
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #launch model
    sess = tf.InteractiveSession()

    #initialize variables
    tf.global_variables_initializer().run()

    #train
    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    #test model after training
    #argmax(y,1) will give the label our model believes is correct. tf.argmax(y_, 1) will give correct label.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

    #cast to floating point to get percentages
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#initialize weights with small amount of noise for symmetry breaking and prevention of 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#initialize with slightly positive bias to avoid dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#convolutions with stride of 1 and zero padded so that output is same size as input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#'plain old max pooling over 2x2 blocks'
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


tf.app.run(main=main)
    
    
    
    
    
    

    
