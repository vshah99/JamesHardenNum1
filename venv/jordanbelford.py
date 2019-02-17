from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


train_dataset_fp = 'TEST.csv'
test_fp = 'TESTTEST.csv'

col_names = ['1 hr','1 min','1.5 hr','10 day','10 min','100 day','125 day','15 min','150 day','175 day','2 day','2 hr','2.5 hr','20 day','20 min','200 day','25 min','3 hr','3.5 hr','30 day','30 min','4 hr','4.5 hr','40 day','45 min','5 day','5 hr','5 min','5.5 hr','50 day','50 min','55 min','6 hr','6.5 hr',
'60 day','70 day','80 day','90 day', 'Change']

lab_names = col_names[-1]
feat_names = col_names[:-1]
class_names = ['DOWN', 'UP']

batch_size = 25
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=col_names,
    label_name=lab_names,
    num_epochs=1
    )

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=col_names,
    label_name=lab_names,
    num_epochs=1,
    shuffle=False)

features, labels = next(iter(train_dataset))

#print(features)

plt.scatter(features['50 day'].numpy(),
            features['200 day'].numpy(),
            c=labels.numpy(),
            cmap='viridis')

plt.xlabel("50 day")
plt.ylabel("200 day")
#plt.show()

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))

test_dataset = test_dataset.map(pack_features_vector)

#print(features[:5])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(38,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

predictions = model(features)
#print(tf.nn.softmax(predictions[:25], axis=1))

#print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#print("    Labels: {}".format(labels))

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
#print("Loss test: {}".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)

'''

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))
'''

## Note: Rerunning this cell uses the same model variables

from tensorflow import contrib

tfe = contrib.eager

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 5

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())



    print(epoch)
    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
