# conv fc1 fc2
from PIL import Image
import numpy  as np
import tensorflow as tf
import cv2 
import time

INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 8
CONV1_SIZE = 40
CONV2_DEEP = 8
CONV2_SIZE = 40
FC_SIZE = 32

batch_size = 8
learning_rate_base = 0.0001
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99

print('-' * 30)
print('Loading train data and Initlizing CNN Network')
print('-' * 30)
train_data = np.load("train_data.npy")
train_label = np.load("train_label.npy")
print('-' * 30)
print('Notice:train_data shape:' + str(train_data.shape))
print('Notice:train_label shape:' + str(train_label.shape))
print('-' * 30)
train_data = train_data.astype('float32')
train_label = train_label.astype('float32')
print('-' * 30)
print('Load train data and Initlize CNN Network Completed')
print('-' * 30)

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1',reuse=tf.AUTO_REUSE):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('layer1-conv1-weights', conv1_weights)
        tf.summary.histogram('layer1-conv1-biases', conv1_biases)

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 10, 10, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        pass
    
    pool_shape = relu1.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(relu1, [pool_shape[0], nodes])

    with tf.variable_scope('layer2-fc1',reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
            pass
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer2-fc1-weights', fc1_weights)
        tf.summary.histogram('layer2-fc1-biases', fc1_biases)
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
        pass

    with tf.variable_scope('layer3-fc2',reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, NUM_LABELS],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
            pass
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer3-fc2-weights', fc2_weights)
        tf.summary.histogram('layer3-fc2-biases', fc2_biases)
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        pass

    return logit

x = tf.placeholder(tf.float32, [8, 480, 640,1], name='x-input')
y_ = tf.placeholder(tf.float32, [8, 10], name='y-input')
regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
y = inference(x, train=True, regularizer=regularizer)
global_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
tf.summary.scalar('loss', loss)
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 8, learning_rate_decay)
tf.summary.scalar('learning_rate', learning_rate)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
with tf.control_dependencies([train_step, variable_averages_op]):
    train_op = tf.no_op(name='train')
    pass
saver = tf.train.Saver()
merged = tf.summary.merge_all()

train_data = np.reshape(train_data,(125,2457600))
train_label = np.reshape(train_label,(125,8,10))

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    train_writer = tf.summary.FileWriter('train4', sess.graph)
    for i in range(training_steps):
        xs = train_data[i%124]
        ys = train_label[i%124]
        xs = np.reshape(xs,(8,480,640,1))
        ys = np.reshape(ys,(8,10))
        _, loss_value, step, summary, accuracy_count = sess.run([train_op, loss, global_step, merged,accuracy],
                                                        feed_dict={x: xs, y_: ys})
        if i % 2 == 0:
            print("After %d training step(s), loss on training batch is %g,accuracy on training batch is %g" % (step, loss_value,accuracy_count))
            saver.save(sess, "train4/savesample.ckpt", global_step=global_step)
            train_writer.add_summary(summary, i)
            train_writer.flush()
            pass
        pass
    train_writer.close()
    pass
