# Handwritten-numeral-recognition
This is a project about using CNN to realize handwritten numeral recognition

---
## How to use

### Dependencies
* Tensorflow(python3)
* Opencv for python

### Data

Write each number from 0 to 9 a hundred times to prepare the data. Then run get_picture.py to capture these numbers into computer.

Then run convert_data_*.py to compress the data and make the data. The train_data.npy or train_data.tfrecords should be generaterd.

### Train

Run train_v*.py to train te model.

### Test

Run eval_v*.py to get the result.
