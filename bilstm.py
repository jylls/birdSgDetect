'''
Created on 11 Jan 2017

@author: jl10015
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# Import MNIST data



dataset1=np.genfromtxt('fmel28280.txt',dtype=float, unpack=True)  
labelset1=np.genfromtxt('label0.txt', dtype=float,unpack=True) 




dataset2=np.genfromtxt('fmel28281.txt',dtype=float, unpack=True)  
labelset2=np.genfromtxt('label1.txt', dtype=float,unpack=True) 




data1= dataset1.transpose()


data2= dataset2.transpose()
print data1.shape, data2.shape

labs1=np.zeros((7690,2))
labs2=np.zeros((8000,2))


for i in xrange(7690):
       
        if labelset1[i]==0:
            labs1[i,0]=1
            labs1[i,1]=0
    
        else:
            labs1[i,0]=0
            labs1[i,1]=1
            
for i in xrange(8000):
   

    if labelset2[i]==0:
        labs2[i,0]=1
        labs2[i,1]=0
    else:
        labs2[i,0]=0
        labs2[i,1]=1


data=np.concatenate((data1, data2), axis=0)
print data.shape


labs=np.concatenate((labs1, labs2), axis=0)
#print labs



ord=np.random.permutation(15690)

data=data[ord]
labs=labs[ord]

labstr=labs[0:15000]
datatr=data[0:15000]
datate=data[15000:15690]
labste=labs[15000:15690]
# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 
n_steps = 28 
n_hidden = 128 # hidden layer num of features
n_classes = 2 

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):

    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        rn=np.random.randint(15000,size=batch_size)
        batch_x, batch_y = datatr[rn],labstr[rn] 
       
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 300
    test_data =datate[:test_len].reshape((-1, n_steps, n_input))
    test_label = labste[:test_len]
    print("Testing Accuracy:", \
sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
