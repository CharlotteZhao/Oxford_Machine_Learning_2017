#### Machine Learning Practical 3

```
test accuracy 0.9724
```



![Unknown](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/Unknown.png)

![2](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/2.png)

![3](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/3.png)

![4](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/4.png)

![5](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/5.png)

![6](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/6.png)

![7](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/7.png)



```

%matplotlib notebook

from matplotlib import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
x_ = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the first convolution layer here
# TODO
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
W_conv1 = weight_variable([12, 12, 1, 25])
b_conv1 = bias_variable([25])

h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


def conv2d_2(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,2,1], padding='SAME')


# Define the second convolution layer here
W_conv2 = weight_variable([5, 5, 25, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d_2(h_pool1, W_conv2) + b_conv2)


# Define maxpooling
h_pool2 = max_pool_2x2(h_conv2)

# All subsequent layers will be fully connected ignoring geometry so we'll flatten the layer
# Flatten the h_pool2_layer (as it has a multidimensiona shape) 
print(h_pool2.shape)
h_pool2_flat = tf.reshape(h_pool2, [-1, 3*2*64])
print(h_pool2_flat.shape)

# Define the first fully connected layer here
W_fc1 = weight_variable([3*2*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Use dropout for this layer (should you wish)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# The final fully connected layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# We'll use the cross entropy loss function 
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# And classification accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# And the Adam optimiser
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# Load the mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Let us visualise the first 16 data points from the MNIST training data

fig = plt.figure()
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(mnist.train.images[i].reshape(28, 28), cmap='Greys_r')  
    
    # Start a tf session and run the optimisation algorithm
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        #print(batch[1])
        train_accuracy = accuracy.eval(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# Print accuracy on the test set
print('test accuracy %g' % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    
# Visualise the filters in the first convolutional layer
with sess.as_default():
    W = W_conv1.eval()
    
    # Add code to visualise filters here
    
    fig = plt.figure()
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(W[:,:,0,i], cmap='Greys_r')
        
        H = sess.run(h_conv1, feed_dict={x: mnist.test.images})

t = np.zeros((10000,9,9,1))
for i in range (0,10000):
    for j in range (0,9):
        for k in range (0,9):
            #t[i,j,k] = H[i,j,k,0] +  H[i,j,k,1] + H[i,j,k,2] + H[i,j,k,3] + H[i,j,k,4] 
            t[i,j,k] = H[i,j,k,24]
sortedt = np.argsort(t,axis=None)
#print(sortedt.shape)
#print(sortedt[0],sortedt[1])
t2 = np.unravel_index(sortedt,(10000,9,9))
length = len(t2[0])
index = np.zeros((12,3))
for i in range (0,12):
    index[i][0] = t2[0][length-i-1]
    index[i][1] = t2[1][length-i-1]
    index[i][2] = t2[2][length-i-1]
im = np.zeros(((12,12,12)))
#print(mnist.test.images.shape)
for i in range (0,12):
    num = int(index[i][0])
    row = int(index[i][1] * 2)
    column = int(index[i][2] * 2)
    temp = np.zeros((28,28))
    temp2 = np.zeros((28,28))

    temp2 = mnist.test.images[num]
    #print(temp2.type)
    temp = temp2.reshape(28,28)
    im[i,:,:] = temp[row:(row+12),column:(column+12)]
    
fig = plt.figure()
for i in range(12):
    ax = fig.add_subplot(4, 3, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(im[i], cmap='Greys_r') 

# Start a tf session and run the optimisation algorithm
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        #print(batch[1])
        train_accuracy = accuracy.eval(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# Print accuracy on the test set
print('test accuracy %g' % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # Visualise the filters in the first convolutional layer
with sess.as_default():
    W = W_conv1.eval()
    
    # Add code to visualise filters here
    
    fig = plt.figure()
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(W[:,:,0,i], cmap='Greys_r')

H = sess.run(h_conv1, feed_dict={x: mnist.test.images})

t = np.zeros((10000,9,9,1))
for i in range (0,10000):
    for j in range (0,9):
        for k in range (0,9):
            t[i,j,k] = H[i,j,k,24]
sortedt = np.argsort(t,axis=None)
t2 = np.unravel_index(sortedt,(10000,9,9))
length = len(t2[0])
index = np.zeros((12,3))
for i in range (0,12):
    index[i][0] = t2[0][length-i-1]
    index[i][1] = t2[1][length-i-1]
    index[i][2] = t2[2][length-i-1]
im = np.zeros(((12,12,12)))
for i in range (0,12):
    num = int(index[i][0])
    row = int(index[i][1] * 2)
    column = int(index[i][2] * 2)
    temp = np.zeros((28,28))
    temp2 = np.zeros((28,28))

    temp2 = mnist.test.images[num]
    temp = temp2.reshape(28,28)
    im[i,:,:] = temp[row:(row+12),column:(column+12)]
    
fig = plt.figure()
for i in range(12):
    ax = fig.add_subplot(4, 3, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(im[i], cmap='Greys_r') 

```

##### optional

filter size:

first layer: 6*6

second layer: 3*3

```
test accuracy 0.9672
```

![6_6](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/6_6.png)

![8](/Users/charlottezhao/Dropbox/Machine Learning/Practical/practical3/8.png)