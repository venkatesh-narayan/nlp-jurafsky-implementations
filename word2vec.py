import numpy as np
import string
import os
import random
import argparse
import gc
from logistic_regression import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, help='corpus location')

args = parser.parse_args()

class word2vec_LR(LogisticRegression):
	def forward(self, x):
		xx = np.expand_dims(x, -1)
		intermediate = xx * self.weights # will be mostly 0s except for 2 weight vectors per elem in batch
		intermediate = intermediate[~np.all(intermediate == 0, axis=-1)] # remove zero-rows

		out = np.array([np.dot(intermediate[i], intermediate[i + 1]) for i in range(0, len(intermediate), 2)])
		return self.sigmoid(out)
	
	def loss(self, pos_pred, neg_preds):
		pos = np.log(pos_pred)
		neg = np.sum(np.log(neg_preds), axis=0)

		return -(pos + neg)
	
	def grad_pos(self, pos_pred, w):
		return (pos_pred - 1) * w
	
	def grad_neg(self, neg_preds, w):
		return neg_preds * w
	
	def grad_w(self, pos_pred, neg_preds, pos, negs):
		return (pos_pred - 1) * pos + np.sum(neg_preds * negs, axis=0)
	
	def sgd(self):
		for x_batch, y_batch in self.train_batches:
			# batches are made in a way where first is pos and rest are neg
			# find index of curr word and pos and neg contexts so that we can update accordingly
			pos = x_batch[0]
			locs = np.where(pos == 1)
			w = locs[0]
			context_pos = locs[1]

			negs = x_batch[1:]
			context_negs = np.array([np.where(neg == 1)[-1] for neg in negs])
			
			# get pos and neg preds
			pos_pred = self.forward(pos)
			neg_preds = self.forward(negs)
			
			# update
			self.weights[context_pos] -= self.lr * self.grad_pos(pos_pred, self.weights[w])
			self.weights[context_negs] -= self.lr * self.grad_neg(neg_preds, self.weights[w])
			self.weights[w] -= self.lr * self.grad_w(pos_pred, neg_preds, 
													 self.weights[context_pos], self.weights[context_negs])			

	def train(self):
		for epoch in range(self.num_epochs):
			self.sgd()

# skip gram with negative sampling model
class word2vec:
	def __init__(self, corpus, alpha=0.75, context=2, k=2, d=300, max_examples_per_word=5):
		self.corpus = corpus # path to corpus
		self.alpha = alpha # weighted sampling for noise words
		self.k = k # ratio of neg to pos examples
		self.context = context # context window for +ve examples
		self.max_examples_per_word = max_examples_per_word
		
		X_train, y_train, X_val, y_val = self.create_datasets()
		self.model = word2vec_LR(X_train, y_train, X_val, y_val, 300, batch_size=k + 1)
		
		print('training model...')
		self.model.train()
	
	def similarity(self, word1, word2):
		idx1 = self.vocabulary[word1]
		idx2 = self.vocabulary[word2]
		
		# get corresponding weights; add context weight to target weight
		weight1 = (self.model.weights[idx1] + 
				   self.model.weights[len(self.vocabulary) + idx1])
		weight2 = (self.model.weights[idx2] + 
				   self.model.weights[len(self.vocabulary) + idx2])
		
		# normalize
		w1 = weight1 / np.linalg.norm(weight1)
		w2 = weight2 / np.linalg.norm(weight2)
		return np.dot(w1, w2)

	def create_datasets(self):
		print('getting vocabulary...')
		self.get_vocabulary()

		print('done! getting positive words...')
		self.get_positive_words()

		# get num_context_words x k randomly sampled noise words for each word
		print('done! getting noise words...')
		self.negatives = { k: self.get_noise_words(k, min(self.max_examples_per_word, len(v))) 
						   for k, v in self.positives.items() }

		# shuffle data to get random split; 80% train 20% val
		idxs = list(range(len(self.vocabulary)))
		random.shuffle(idxs)
		split_point = int(0.8 * len(idxs))
		
		def split(idxs, start, end):
			X, y = [], []
			for idx in idxs[start:end]:
				# convert positives into one-hot vectors of size 2|V|
				# there will be two 1's: one at idx, and the other at positives[idx][i]
				# corresponding negative labels will be at negatives[idx][k*i:k*(i + 1)]
				# if doing for all context words, can kill memory; just do it for at most
				# max_examples_per_word words
				total = min(self.max_examples_per_word, len(self.positives[idx]))
				for i in range(total):
					# splitting it into two halves and then putting them together later (easier to index into)
					pos1, pos2 = [0] * len(self.vocabulary), [0] * len(self.vocabulary)
					pos1[idx] = 1; pos2[self.positives[idx][i]] = 1
					
					# for each positive label, there are k negative labels --> 2k negative halves
					# go by increments of 2; first half will always have 1 at idx, second half will
					# have 1 somewhere between k * i and k*(i + 1), depending on where we are in the
					# list; this is tracked by counter
					negs = [[0] * len(self.vocabulary) for _ in range(2 * self.k)]
					negs_cat = []
					counter = self.k * i
					for j in range(0, 2 * self.k, 2):
						negs[j][idx] = 1
						negs[j + 1][self.negatives[idx][counter]] = 1
						
						# concatenate negative examples
						negs_cat.append(negs[j] + negs[j + 1])

						counter += 1
					
					# concat pos example and give label as 1
					X.append(pos1 + pos2)
					y.append(1)

					# add neg concatenated examples and give label as 0
					for neg in negs_cat:
						X.append(neg)
						y.append(0)
					
					# save memory
					del pos1
					del pos2
					del negs
					del negs_cat

			gc.collect()
				
			# turn into np arrays
			X = np.array(X)
			y = np.array(y)	
			return X, y

		print('done! splitting...')
		X_train, y_train = split(idxs, 0, split_point)
		X_val, y_val = split(idxs, split_point, len(idxs))
		
		print('done!')
		return X_train, y_train, X_val, y_val

	def get_vocabulary(self):
		vocabulary = set()
		unigram_counts = dict()

		with open(self.corpus, 'r') as f:
			for line in f:
				# get all words and add all non-punctuation words to vocab
				# find unigram counts of all words simultaneously
				words = line.split(' ')
				for word in words:
					if word not in string.punctuation:
						vocabulary.add(word)
						unigram_counts[word] = (1 if word not in unigram_counts 
												else unigram_counts[word] + 1)

		vocabulary = list(vocabulary) # convert to list for indexing
		self.vocabulary = { v: i for i, v in enumerate(vocabulary) } # convert to dict
		
		# make keys into indexes 
		self.unigram_counts = { self.vocabulary[k]: v for k, v in unigram_counts.items() }

	
	def get_positive_words(self):
		self.positives = dict() # list of positive context words for each word in vocab

		with open(self.corpus, 'r') as f:
			text = f.read()
		
		# get all words, and remove punctuation tokens
		# find the context words for each word and add to dict
		words = [x for x in text.split(' ') if x not in string.punctuation]
		for i in range(len(words)):
			start = max(0, i - self.context)
			end = min(i + self.context, len(words))
			context_words = words[start:i] + words[i + 1:end] # skip curr word
			context_words = [self.vocabulary[word] for word in context_words] # conver to idxs
			
			# add all context words (by index) to dict entry for curr word
			curr_word = words[i]
			if self.vocabulary[curr_word] in self.positives:
				self.positives[self.vocabulary[curr_word]].extend(context_words)
			else:
				self.positives[self.vocabulary[curr_word]] = context_words

	def get_noise_words(self, curr_word, repeat):
		# noise words are random word from lexicon that's not the curr word
		# found by weighting unigram probability --> gives rarer words more prob mass
		# first find sum of (counts of all words (except curr) raised to alpha)
		# then compute count ^ alpha / total for each word (except curr) 
		counts = []
		for k, v in self.unigram_counts.items():
			if k == curr_word:
				counts.append(0) # don't assign any probability mass to the curr word
			else:
				counts.append(v)
		
		counts = np.power(counts, self.alpha)
		total = sum(counts)
		probs = counts / total

		# get random idxs given probability
		chosen = np.random.choice(list(range(len(self.vocabulary))), self.k * repeat, p=probs)
		return chosen
	
			
w2v = word2vec(args.corpus_path)
while True:
	i1 = input('word 1: ')
	i2 = input('word 2: ')

	print(w2v.similarity(i1, i2))


