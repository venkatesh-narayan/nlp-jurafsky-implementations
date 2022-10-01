import numpy as np
import argparse
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help='location of train txt file')
parser.add_argument('--test_file', type=str, help='location of test txt file')
parser.add_argument('--binarize', action='store_true', help='whether or not to do binarize naive bayes')

args = parser.parse_args()

# given a txt file, create the docs (format described above train_nb)
# assuming each line of file follows: [class_name] [text]
def get_docs(fname, mode):
	docs = []
	with open(fname, 'r') as f:
		for line in f:
			if mode == 'train':
				tokens = line.split(' ')
				docs.append([' '.join(tokens[1:]), tokens[0]])
			else:
				docs.append(line)
	
	return docs

# expecting docs to be of the form [[doc1, class1], [doc2, class2], ...]
def train_nb(docs, binarize, alpha=1):
	# get unique classes
	classes = list(set([label for _, label in docs]))
	total_classes_size = len(classes)
	
	# get vocabulary -- required for both train for test
	vocabulary = set()
	for doc, _ in docs:
		# not doing any fancy tokenization -- just plain word splits
		for word in doc.split(' '):
			# remove any newlines at the end
			vocabulary.add(word.rstrip('\n'))
	
	vocabulary = { word: index for index, word in enumerate(list(vocabulary)) } # convert to dict
	
	logprior = np.zeros((total_classes_size,)) # log prior probabilities per class
	loglikelihood = np.zeros((len(vocabulary), total_classes_size)) # loglikelihood of word given class
	for i, c in enumerate(classes):
		num_docs = len(docs)
		relevant_docs = [doc for doc, label in docs if label == c] # get all c-labeled docs
		num_c = len(relevant_docs)
		
		# compute prior probabilities per class (num documents in class c / num documents)
		prior = num_c / num_docs
		logprior[i] = math.log(prior)

		# get counts of words from relevant docs
		relevant_vocabulary = { k: 0 for k in vocabulary.keys() }
		for doc in relevant_docs:
			# if binary, then just remove the repeats within docs
			words = list(set(doc.split(' '))) if binarize else doc.split(' ')
			for word in words:
				# remove any newlines from end
				relevant_vocabulary[word.rstrip('\n')] += 1

		for word, count in relevant_vocabulary.items():
			# calculate p(word | class) with smoothing
			idx = vocabulary[word]
			likelihood = (count + alpha) / (len(relevant_vocabulary) + len(vocabulary) * alpha)
			loglikelihood[idx, i] = math.log(likelihood)
	
	return logprior, loglikelihood, classes, vocabulary
			

# expecting docs to be of the form [[doc1, class1], [doc2, class2], ...]
def test_nb(docs, logprior, loglikelihood, classes, vocabulary):
	best_classes = [0] * len(docs)
	for i, doc in enumerate(docs):
		# find the probs for each class for each doc
		logprobs = np.zeros((len(classes),))
		for c in range(len(classes)):
			logprobs[c] = logprior[c]

			for word in doc.split(' '):
				word = word.rstrip('\n')
				# discard words not in vocabulary
				# otherwise, logprobs = logprior + loglikelihood
				if word in vocabulary:
					logprobs[c] += loglikelihood[vocabulary[word], c]
		
		# get the class corresponding to the argmax of all log probs
		best_classes[i] = classes[np.argmax(logprobs)] 
	
	return best_classes
			

train_docs = get_docs(args.train_file, 'train')
logprior, loglikelihood, classes, vocabulary = train_nb(train_docs, args.binarize)

test_docs = get_docs(args.test_file, 'test')
best_classes = test_nb(test_docs, logprior, loglikelihood, classes, vocabulary)
for doc, best_class in zip(test_docs, best_classes):
	print(f'document: {doc}')
	print(f'sentiment: {best_class}')
	print()


