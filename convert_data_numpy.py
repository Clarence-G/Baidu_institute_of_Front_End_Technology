import tensorflow as tf
import numpy as np
from PIL import Image
import os

train_data = np.empty(shape = [1])
train_label = np.empty(shape = [1])

num_pic = 0
for j in range(0,10):
    value = j
    labels = [0] * 10
    labels[int(value)] = 1
    for i in range(1,101):
        current_filename = "xxxxx" + str(j)  + "/" + str(i) + ".jpg"  # write your data path here
        im = Image.open(current_filename)
        im = im.resize((64,48))
        width,height = im.size
        if(height != 48 or width != 64):
            print("Error:Image Size Not Match Expected 64*48 Reveived " + str(height) + "*" + str(width))
            sys.exit(1)
            pass
        image = im.convert("I")
        data = image.getdata()
        data = np.matrix(data)
        new_data = np.reshape(data,(48*64))
        train_data = np.append(train_data,new_data)
        train_label = np.append(train_label,labels)
        num_pic +=1
        print("the number of picture:{}".format(num_pic))
        if(i == 1 and j == 0):
            train_data = np.delete(train_data,[0])
            train_label = np.delete(train_label,[0])
            pass
        pass

    pass


np.save('train_data_v2.npy', train_data)
np.save('train_label_v2.npy', train_label)

print(str(train_data.shape))
print(str(train_data.dtype))
print(str(train_label.shape))
print("Done.")