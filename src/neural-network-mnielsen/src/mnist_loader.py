### import the libraries
import pickle
import gzip

def load_data():
	"""return the MNIST data as a tuple of training data, 
	validation data, and test data"""
	f = gzip.open('../data/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
	f.close()
	return training_data, validation_data, test_data

def load_data_wrapper():
	"""return a formatted data from load_data, for use in
	neural network code implementation"""
	tr_d, va_d, te_d = load_data()
	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_results = [vectorized_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
	test_data = zip(test_inputs, te_d[1])
	return training_data, validation_data, test_data

def vectorized_result(j):
	"""return a 10-dimensionalunit vector with position of j has
	value of 1.0"""
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

 