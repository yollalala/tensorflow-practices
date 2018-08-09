### basic operation with eager mode
# import the needed libraries
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# set the eager API
tfe.enable_eager_execution()
# define the constants
a = tf.constant(12)
print('a = %d' % a)
b = tf.constant(32)
print('b = %d' % b)
# operation running without session
c = a + b
print('a + b = %d' % c)
d = a * b
print('a * b = %d' % d)
# mixing operation with tensor and numpy arrays
a = tf.constant([[2., 1.],
				 [1., 0.]], dtype=tf.float32)
print('Tensor a:')
print('%s' % a)
b = np.array([[3., 0.],
			  [5., 1.]], dtype=np.float32)
print('Numpy Array b:')
print('%s' % b)
# operation running
c = a + b
print('a + b = %s' % c)
d = tf.matmul(a, b)
print('a * b = %s' % d)
# Tensor iteration
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		print(a[i][j])

