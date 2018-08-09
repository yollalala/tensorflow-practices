### linear regression with one neuron
### import the needed libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# constants
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# train sets
X_train = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
	7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y_train = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
	2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = X_train.shape[0]

# graph input
X = tf.placeholder('float')
Y = tf.placeholder('float')

# initialize model's weights and biases
W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

# construct a linear model, y = X.W + b
model = tf.add(tf.multiply(X, W), b)

# mean squared error
cost = tf.reduce_sum(tf.pow(model - Y, 2)) / (2 * n_samples)
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# get the variable's initializer
init = tf.global_variables_initializer()

# start the training
with tf.Session() as sess:
	sess.run(init)
	# fit the train sets
	for epoch in range(training_epochs):
		for (x, y) in zip(X_train, Y_train):
			sess.run(optimizer, feed_dict={X:x, Y:y})
		# display logs per 50 epochs
		if(epoch + 1) % display_step == 50:
			c = sess.run(cost, feed_dict={X:X_train, Y:Y_train})
			print('Epoch:', '%04d' % (epoch+1), 'cost=', '%.9f' % c, 
				'W= %f' % sess.run(W),'b= %f' % sess.run(b))

	print('Training finished')
	training_cost = sess.run(cost, feed_dict={X:X_train, Y:Y_train})
	print('Training cost=', training_cost, 'W=', sess.run(W), 'b=', sess.run(b))

	# result's visualization
	plt.plot(X_train, Y_train, 'ro', label='Original data')
	plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label='Fitted line')
	plt.legend()
	plt.show()







