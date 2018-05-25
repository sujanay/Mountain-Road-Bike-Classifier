
# coding: utf-8

# # Using Different Train and Test File
# # TensorFlow and Keras to build CNN Image classifier

# In[1]:


import math
import os
from six.moves import xrange
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import itertools
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


# We have only 2 classes - Mountain bike and Road bike
NUM_CLASSES = 2

# The training images are processed converting them to 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 10        # Total training images = 140

# Maximum number of training steps.
MAX_STEPS = 2000

# Directory to put the training data.
TRAIN_DIR='mountain-and-road/train'


# In[3]:


train_batches = ImageDataGenerator().flow_from_directory(TRAIN_DIR, target_size=(28,28), classes=['mountain','road'], batch_size=10)


# In[4]:


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1]!=3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) //rows if len(ims)%2 ==0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# In[5]:


def cnn_inference(X):

    K=4
    L=8
    M=12

    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))
    B1 = tf.Variable(tf.ones([K])/10)
    W2 = tf.Variable(tf.truncated_normal([4, 4, 4, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L])/10)
    W3 = tf.Variable(tf.truncated_normal([4, 4, 8, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M])/10)
    
    N=200

    W4 = tf.Variable(tf.truncated_normal([7*7*M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N])/10)
    W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=0.1))
    B5 = tf.Variable(tf.zeros([2])/10)
    
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)  
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)

    YY = tf.reshape(Y3, shape=[-1, 7*7*M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    Y  = tf.nn.softmax(tf.matmul(Y4, W5) + B5)
    
    return Y


# In[6]:


def cnn_training(y_, Y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=Y))
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(1e-4)
    
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
        
    return train_op, loss


# In[7]:


cnn_graph = tf.Graph()
with cnn_graph.as_default():
    # Generate placeholders for the images and labels.
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])                                      
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    tf.add_to_collection("images", X)  # Remember this Op.
    tf.add_to_collection("labels", y_)  # Remember this Op.

    # Build a Graph that computes predictions from the inference model.
    Y = cnn_inference(X)
    tf.add_to_collection("Y", Y)  # Remember this Op.

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op, loss = cnn_training(y_, Y)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()


# In[11]:


with tf.Session(graph=cnn_graph) as sess:
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(MAX_STEPS):
        # Read a batch of images and labels.
        imgs, labels = next(train_batches)

        _, loss_value = sess.run([train_op, loss],
                                 feed_dict={X: imgs,
                                            y_:labels})

        losses.append(loss_value)
        # Print out loss value.
        if step % 1000 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))

    # Write a checkpoint.
    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)

