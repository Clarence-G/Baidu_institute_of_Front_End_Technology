# fc1 fc2
from PIL import Image
import numpy  as np
import tensorflow as tf
import cv2 
import time

NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 3
CONV2_DEEP = 32
CONV2_SIZE = 3
FC_SIZE = 128
nodes = 3072
batch_size = 8
learning_rate_base = 0.0001
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99

print('-' * 30)
print('Loading train data and Initlizing CNN Network')
print('-' * 30)

print('-' * 30)
print('Loading train data and Initlizing CNN Network')
print('-' * 30)
train_data = np.load("train_data_v2.npy")
train_label = np.load("train_label_v2.npy")
print('-' * 30)
print('Notice:train_data_v2 shape:' + str(train_data.shape))
print('Notice:train_label_v2 shape:' + str(train_label.shape))
print('-' * 30)
train_data = train_data.astype('float32')
train_label = train_label.astype('float32')
print('-' * 30)
print('Load train data and Initlize CNN Network Completed')
print('-' * 30)
train_data = np.reshape(train_data,(1000,48*64))
train_label = np.reshape(train_label,(1000,10))

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
dataset = dataset.shuffle(buffer_size=100000)
dataset = dataset.batch(8)
#dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

def inference(input_tensor, train, regularizer):

    with tf.variable_scope('layer1-fc1',reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
            pass
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer1-fc1-weights', fc1_weights)
        tf.summary.histogram('layer1-fc1-biases', fc1_biases)
        fc1 = tf.nn.relu(tf.matmul(input_tensor, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
        pass

    with tf.variable_scope('layer2-fc2',reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, NUM_LABELS],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
            pass
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer2-fc2-weights', fc2_weights)
        tf.summary.histogram('layer2-fc2-biases', fc2_biases)
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        pass

    return logit

x = tf.placeholder(tf.float32, [8, 48*64], name='x-input')
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
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 125, learning_rate_decay)
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


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    train_writer = tf.summary.FileWriter('train3', sess.graph)
    for i in range(training_steps):
        xs,ys = sess.run(next_element)
        #xs = xs.eval()
        #ys = ys.eval()
        xs = np.reshape(xs,(8,48*64))
        ys = np.reshape(ys,(8,10))
        _, loss_value, step, summary, accuracy_count = sess.run([train_op, loss, global_step, merged,accuracy],
                                                        feed_dict={x: xs, y_: ys})
        print("After %d training step(s), loss on training batch is %g,accuracy on training batch is %g" % (step, loss_value,accuracy_count))
        saver.save(sess, "train3/savesample.ckpt", global_step=global_step)
        train_writer.add_summary(summary, i)
        train_writer.flush()
        pass
    train_writer.close()
    pass


