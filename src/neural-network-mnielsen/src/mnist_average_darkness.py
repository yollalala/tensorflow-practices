### import the libraries
from collections import defaultdict
import mnist_loader

def main():
	"""The main script to model and train the model""" 
	# load the data
	training_data, validation_data, test_data = mnist_loader.load_data()
	# training phase: compute the average darkness of each digit
	avgs = avg_darknesses(training_data)
	# testing phase: evaluate the model
	num_correct = sum(int(guess_digit(image, avgs) == digit)
		for image, digit in zip(test_data[0], test_data[1]))
	print('The evaluation results:')
	print('%s of %s values correct.' % (num_correct, len(test_data[1])))

def avg_darknesses(training_data):
	"""Return a defaultdict whose keys are [0..9]"""
	digit_counts = defaultdict(int)
	darknesses = defaultdict(float)
	for image, digit in zip(training_data[0], training_data[1]):
		digit_counts[digit] += 1
		darknesses[digit] += sum(image)
	avgs = defaultdict(float)
	for digit, n in digit_counts.iteritems():
		avgs[digit] = darknesses[digit] / n
	return avgs

def guess_digit(image, avgs):
	"""Return the guess from the model (prediction)"""
	darkness = sum(image)
	distances = {k : abs(v - darkness) for k, v in avgs.iteritems()}
	return min(distances, key=lambda x: distances[x])

# if __name__ == '__main__':
# 	main()
