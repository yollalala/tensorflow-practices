### neural network model for predicting handwritten digits
### using MNIST datasets
### tutorials from tensorflow.org

# load the datasets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/', one_hot=True)

# import tensorflow library
import tensorflow as tf

# graph input definition 
# the placeholder for images, the 784-dimensional vector
X = tf.placeholder(tf.float32, [None, 784])

# variables definition
# for weights W and biases b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define the model using softmax
y = tf.nn.softmax(tf.matmul(X, W) + b)

# implement the cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])	# for the labels
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, 
	logits=y)

# gradient descent
learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# get the session
sess = tf.InteractiveSession()

# initialize the variables
tf.global_variables_initializer().run()

# train the neural network model
# but, instead one epoch for all training sets
# we do one epoch for a mini-batch
training_epochs = 1000
batch_size = 100
for i in range(training_epochs):
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})

# evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('The accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, y_:mnist.test.labels}))





