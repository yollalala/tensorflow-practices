### variables practices
### tutorials from tensorflow documentation

# import the libraries
import numpy as np
import tensorflow as tf

# creating a variable
my_variable = tf.get_variable('my_variable', [1, 2, 3])

# specify the dtype
my_int_variable = tf.get_variable('my_int_variable', [1, 2, 3], 
	dtype=tf.int32, initializer=tf.zeros_initializer)

# initialize Variable with a Tensor
other_variable = tf.get_variable('other_variable', dtype=tf.int32, 
	initializer=tf.constant([23, 42]))

## variable collections
# add a non-trainable variable into collection
my_local = tf.get_variable('my_local', shape=(), 
	collections=[tf.GraphKeys.LOCAL_VARIABLES])
# or alternatively,
my_non_trainable = tf.get_variable('my_non_trainable', shape=(),
	trainable=False)

# add variable to our custom collection
tf.add_to_collection('my_collection', my_local)
# retrieve the collection
print(tf.get_collection('my_collection'))

# device placement
# with tf.device('/device:GPU:1'):
# 	v = tf.get_variable('v', [1])

# automatically place variables in parameter servers
# cluster_spec = {
# 	'ps': ['ps0:2222', 'ps1:2222'],
# 	'worker': ['worker0:2222', 'worker1:2222', 'worker2:2222']
# }
# with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
# 	u = tf.get_variable('u', shape=[20, 20])

## variables initializing
# initialize variable manually
# tf.Session().run(my_int_variable)

# initialize all trainable variables
# tf.Session().run(tf.global_variables_initializer())

# print the unintialized variables
sess = tf.Session()
# print(sess.run(tf.report_uninitialized_variables()))

# initialize a variable where not all variables are initialized
x = tf.get_variable('x', shape=(), initializer=tf.zeros_initializer())
y = tf.get_variable('y', initializer=x.initialized_value() + 1)

## using variables
a = tf.get_variable('a', shape=(), initializer=tf.zeros_initializer())
b = a + 1
# sess.run(tf.global_variables_initializer())
# print(sess.run(b))

# assign a value to a variable
a = tf.get_variable('c', shape=(), initializer=tf.zeros_initializer())
assignment = a.assign_add(1)
sess.run(tf.global_variables_initializer())
print(sess.run(a))
print(sess.run(assignment))

# re-read the value of a variable
d = tf.get_variable('d', shape=(), initializer=tf.zeros_initializer())
assignment2 = d.assign_add(1)
with tf.control_dependencies([assignment2]):
	e = d.read_value()
sess.run(tf.global_variables_initializer())
print(sess.run(e))
print(sess.run(d))





