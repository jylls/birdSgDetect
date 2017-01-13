
'''
Created on 2 Nov 2016

@author: jl10015
'''
import tensorflow as tf
from brian import *
from scipy.ndimage.filters import gaussian_filter


datatest=genfromtxt('fmel2828test.txt',dtype=float, unpack=True) 
dataset1=genfromtxt('fmelspecdenoise28280.txt',dtype=float, unpack=True)  
labelset1=genfromtxt('labelsdelta1.txt', dtype=float,unpack=True) 
print mean(dataset1)



dataset2=genfromtxt('fmelspecdenoise28281.txt',dtype=float, unpack=True)  
labelset2=genfromtxt('labelsdelta2.txt', dtype=float,unpack=True) 




data1= dataset1.transpose()
data2= dataset2.transpose()





datatest=datatest.transpose()
print data1.shape, data2.shape#, datatest.shape
'''
data1=dataset1.reshape((1000,28*28))
data2=data1=dataset2.reshape((1000,28*28))
print data1.shape'''


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


labs=np.concatenate((labs1, labs2), axis=0)
#print labs



ord=np.random.permutation(15690)
#print ord
data=data[ord]
labs=labs[ord]
#labsbis=labs[0:15000]

############################################################################
#labstr=np.concatenate((labs[0:15000],labsbis), axis=0)
#datatr=np.concatenate((data[0:15000], np.fliplr(data[0:15000])), axis=0)
###########################################################################
datatr=data[0:15000]
labstr=labs[0:15000]


datate=data[15000:15690]
labste=labs[15000:15690]



print datatr.shape,labstr.shape

batch_size = 128
test_size=690

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)


    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


trX, trY, teX, teY,testX = datatr, labstr, datate, labste,datatest
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
testX=testX.reshape(-1,28,28,1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])    # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 2])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]


        print(i,np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX,
                                                             p_keep_conv: 1.0,
    p_keep_hidden: 1.0})))
    y_pred = sess.run(predict_op,feed_dict={X: testX, p_keep_conv: 1.0,
    p_keep_hidden: 1.0})   
    f1=open('ytestpred2.txt',"a")
    np.savetxt(f1,y_pred)
    f1.close()
