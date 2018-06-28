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
FC_SIZE1 = 128

batch_size = 8
learning_rate_base = 0.0001
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99

def inference(input_tensor, train, regularizer):
    
    nodes = 480*640
    reshaped = tf.reshape(input_tensor, [1, nodes])

    with tf.variable_scope('layer5-fc1',reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE1],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
            pass
        fc1_biases = tf.get_variable("bias", [FC_SIZE1], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer5-fc1-weights', fc1_weights)
        tf.summary.histogram('layer5-fc1-biases', fc1_biases)
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
        pass

    with tf.variable_scope('layer6-fc2',reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable("weights", [FC_SIZE1, NUM_LABELS],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
            pass
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer6-fc2-weights', fc2_weights)
        tf.summary.histogram('layer6-fc2-biases', fc2_biases)
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        pass

    return logit

x = tf.placeholder(tf.float32, [1, 480, 640,1], name='x-input')
y_ = tf.placeholder(tf.float32, [8, 10], name='y-input')
regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
y = inference(x, train=True, regularizer=regularizer)
global_step = tf.Variable(0, trainable=False)
saver = tf.train.Saver()
merged = tf.summary.merge_all()

cap = cv2.VideoCapture(0)

with tf.Session() as sess:
    module_file =  tf.train.latest_checkpoint('C://Users/lisixu/Desktop/YYG_Final/Our_trin_set/train5/')
    saver.restore(sess, module_file)
    while(1):
        data_temp = []
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        Grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data = np.matrix(Grayimg)
        data = data.astype(np.float32)
        data_temp.append(data)
        data_temp.append(data)
        data_temp = np.reshape(data_temp,(2,1,480,640,1))
        y_predict = sess.run([y],feed_dict={x: data_temp[0]})
        print(np.where(y_predict==np.max(y_predict)))
        #print("Current number:%d" % tf.argmax(y_predict))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    pass