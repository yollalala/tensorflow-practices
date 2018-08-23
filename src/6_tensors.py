### tensors practices
### tutorials from tensorflow documentation

# import the libraries
import tensorflow as tf

# create rank 0 tensors (scalar)
mammal = tf.Variable('Elephant', tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

# create rank 1 tensors (vector)
mystr = tf.Variable(['Hello'], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# create higher ranks tensors
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4],[9],[16],[25]], tf.int32)
squarish_squares = tf.Variable([[4,9],[16,25]], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)
# image representation with rank of 4
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color

# get the tensor's rank
r = tf.rank(my_image)
sess = tf.Session()
print(sess.run(r))

# get an element from tensor rank 2 by index
my_scalar = squarish_squares[:, 1]
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(my_scalar))

# make a vector of zeros with the same size as 
# the number of matrix's column (squarish_squares)
zeros = tf.zeros(squarish_squares.shape[1])
print(sess.run(zeros))

# reshape the tensors
rank_three_tensor = tf.ones([3, 4, 5])	# shape of the tensor 3x4x5
matrix = tf.reshape(rank_three_tensor, [6, 10])	# reshape into 6x10 tensor
matrixA = tf.reshape(matrix, [3, -1])	# reshape into 3x20
matrixB = tf.reshape(matrixA, [4, 3, -1])	# reshape into 4x3x5
#error_matrix = tf.reshape(matrixB, [13, 2, 2])	# error, shape value does not match

# cast a constant integer tensor into float tensor
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)

# inspect the tensor's datatype
print(float_tensor.dtype)

# evaluate tensors
constant = tf.constant([1, 2, 3])
tensor = constant * constant
with sess.as_default():
	print(tensor.eval())	# only work with the default session

# evaluate tensor with placeholder tensor
with sess.as_default():
	p = tf.placeholder(tf.float32)
	t = p + 1.0
	#t.eval()	# will fail
	print(t.eval(feed_dict={p:2.0}))	# succeed

# print tensors value
t = tf.constant([1, 2, 3])
tf.Print(t, [t])	# does nothing
v = tf.Print(t, [t])
v = v + 1
print(sess.run(v))


