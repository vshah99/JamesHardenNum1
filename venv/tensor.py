import tensorflow as tf
import numpy as np
import csv

batch_size = 1
train_dataset_fp = csv.reader('TEST_SAFE.csv')

col_names = ['1 hr','1 min','1.5 hr','10 day','10 min','100 day','125 day','15 min','150 day','175 day','2 day','2 hr','2.5 hr','20 day','20 min','200 day','25 min','3 hr','3.5 hr','30 day','30 min','4 hr','4.5 hr','40 day','45 min','5 day','5 hr','5 min','5.5 hr','50 day','50 min','55 min','6 hr','6.5 hr',
'60 day','70 day','80 day','90 day']
print(len(col_names))

lab_names = ['UP', 'DOWN']

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=col_names,
    label_name=lab_names,
    num_epochs=1
    )



