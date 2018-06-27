import tensorflow as tf
import numpy as np
from PIL import Image
import os


tfrecords_filename = './train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
num_pic = 0
for j in range(0,10):
    value = j
    labels = [0] * 10
    labels[int(value)] = 1
    for i in range(1,101):
        current_filename = "xxxxx" + str(j)  + "/" + str(i) + ".jpg"     #write you data path here
        im = Image.open(current_filename)
        im = im.resize((64,48))
        width,height = im.size
        if(height != 48 or width != 64):
            print("Error:Image Size Not Match Expected 64*48 Reveived " + str(height) + "*" + str(width))
            sys.exit(1)
            pass
        img_raw = im.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature = {
                'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw])),
                'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = labels))
            }))

        writer.write(example.SerializeToString())
        num_pic +=1
        print("the number of picture:{}".format(num_pic))
        pass

    pass
writer.close()
print("Done.")