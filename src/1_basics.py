import tensorflow as tf

### print byte object of 'Hello World!'
hello = tf.constant('Hello')
world = tf.constant('World')
sess = tf.Session()
print(sess.run(hello + ' ' + world + '!'))

### basic constant operation
a = tf.constant(2)
b = tf.constant(3)
# launch the default graph
with tf.Session() as sess:
	print('a + b = %d' % sess.run(a + b))
	print('a - b = %d' % sess.run(a - b))
	print('a * b = %d' % sess.run(a * b))
	print('a / b = %f' % sess.run(a / b))

### basic graph input operation
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
c = tf.placeholder(tf.int16)
d = tf.constant(3)
e = tf.constant(10)
# define operations
add = tf.add(a, b)
add2 = tf.add(d, e)
subtract = tf.subtract(a, b)
multiply = tf.multiply(a, b)
divide = tf.divide(a, b)
# launch the default graph
with tf.Session() as sess:
	print('a - b = %d' % sess.run(add, feed_dict={a:2, b:4, c:5}))
	print('d + e = %d' % sess.run(add2))
	print('a - b = %d' % sess.run(subtract, feed_dict={b:1, a:3}))
	print('c * a = %d' % sess.run(multiply, feed_dict={b:10, a:2, c:10}))
	print('b / c = %f' % sess.run(divide, feed_dict={b:30, c:4, a:5}))

### matrix operation
# create an 1x2 matrix
matrix1 = tf.constant([[3., 3.]])
# create an 2x1 matrix
matrix2 = tf.constant([[2.],[2.]])
# define operation of matrix multiplication 
matrix_multiply = tf.matmul(matrix1, matrix2)
# launch the graph
with tf.Session() as sess:
	result = sess.run(matrix_multiply)
	print('Matrix multiplication result:')
	print(result)



	
