### tensorflow's introduction practices
### tutorials from tensorflow documentation

# import the libraries
import tensorflow as tf
import numpy as np

# build the simple computational graph
# define the constants
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)	# implicitly defined tf.float32
total = a + b
print(a)
print(b)
print(total)	# doesn't output the value, but the Tensor object

## save the computational graph to a TensorBoard summary file
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())

# create the session and run the computational graph
sess = tf.Session()
print(sess.run(total))

# run the computational graph in dictionaries (multioperations)
print(sess.run({'ab':(a, b), 'total':total}))

# single-valued tensor
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

# create the inputs graph (placeholder)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
# run the graph using input parameter
print(sess.run(z, feed_dict={x:3, y:4.5}))
print(sess.run(z, feed_dict={x:[1, 3], y:[2, 4]}))

# process datasets with iterator
my_data = [[0, 1], [2, 3], [4, 5], [6, 7]]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
# print the item (each row)
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
    
# process datasets with initialize the iterator
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()
# print the item (each row)
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

# create layers
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
# initialize the layers with initializer
init = tf.global_variables_initializer()
sess.run(init)
# execute the layers
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

# create layer function shortcuts
x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.layers.dense(x, units=1)
# initialize the layers
init = tf.global_variables_initializer()
sess.run(init)
# execute the layers
print(sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]}))

# feature columns
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}
department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)
columns = [
    tf.feature_column.numeric_column('sales'),
    department_column ]
inputs = tf.feature_column.input_layer(features, columns)
# initialize the feature columns
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()    # for categorical column initializing
sess.run((var_init, table_init))
# show the one-hot coding (for categorical) and the numeric column
print(sess.run(inputs))

## train a simple regression model
# define the data
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
# define the model
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)
## execute the not-yet-trained prediction
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(y_pred))
# define the loss function (mse)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
# print(sess.run(loss))
# train the model with gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# execute the training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)
print(sess.run(y_pred))









