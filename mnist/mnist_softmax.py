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
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    

    
