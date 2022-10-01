import argparse
import string
import time
import math

import numpy as np

from nltk.tokenize import word_tokenize


parser = argparse.ArgumentParser()

parser.add_argument('--corpus_path', type=str, help='path to corpus')
parser.add_argument('--N', type=int, help='N-gram amount')
parser.add_argument('--vocab_size', type=int, default=50000, help='size of vocabulary')
parser.add_argument('--smoothing', type=str, default='laplace', help='type of smoothing to apply')
parser.add_argument('--k', type=float, default=0.5, help='for add-k smoothing')
parser.add_argument('--num_words_generate', type=int, default=10, help='number of words to generate')

args = parser.parse_args()

class NGram:
	def __init__(self, corpus, N, vocab_size, smoothing):
		assert N > 0, 'need positive value of N'

		with open(corpus, 'r') as f:
			text = f.read()

		tokens = word_tokenize(text) # assuming text is somewhat normal (ends with periods, doesn't have weird characters)
		
		# generate roughly 80-20 split
		split = int(0.8 * len(tokens))
		closest_punctuation, check = 0, float('inf')
		for i in range(len(tokens)):
			# ensure it's not the last token
			if tokens[i] in string.punctuation and abs(split - i) < check and i != len(tokens) - 1:
				closest_punctuation = i
				check = abs(split - i)
		
		# save tokens in sentences and ensure that we start with SOS token and end with EOS token
		# for N-grams, we need to add more SOS tokens in the beginning to calculate probabilities
		# properly
		self.train_tokens = ['<s>'] * (N - 1)
		self.val_tokens = []
		
		L = self.train_tokens # list that we'll append tokens to; initially, this will be train
		
		# wherever there's any punctuation, add EOS and then SOS
		for i, token in enumerate(tokens):
			if token in string.punctuation:
				if i == closest_punctuation:
					L.append('</s>') # add last eos to train

					L = self.val_tokens   		 # switch to val tokens
					L.extend(['<s>'] * (N - 1))  # add SOS enough times

				else:
					# just normally add EOS and SOS
					L.append('</s>')
					L.append('<s>')
			else:
				L.append(token)
		
		self.N = N
		
		# sort all words by their frequency
		unique, counts = np.unique(self.train_tokens, return_counts=True)
		idxs = counts.argsort()
		sorted_unique = unique[idxs[::-1]]

		# take the first vocab_size - 1 most frequent elements and then rest are <UNK>
		self.vocab = list(sorted_unique[:min(len(sorted_unique), vocab_size - 1)]) + ['<UNK>']
		self.vocab_size = len(self.vocab)

		# change train and val accordingly
		change = []
		for token in self.train_tokens:
			if token not in self.vocab:
				change.append('<UNK>')
			else:
				change.append(token)

		self.train_tokens = change
		
		change = []
		for token in self.val_tokens:
			if token not in self.vocab:
				change.append('<UNK>')
			else:
				change.append(token)

		self.val_tokens = change

		# save memory
		del change

		self.smoothing = smoothing
		
		print('computing ngram probs...')
		start = time.time()
		self.compute_ngram_probs()
		print(f'done! took {time.time() - start}s')
	
	def compute_ngram_probs(self):
		self.gram_count = dict() # store N-grams, (N - 1)-grams, and their counts
		
		# start at n - 1 because that's where the first real token is
		for i in range(self.N - 1, len(self.train_tokens)):
			# p(wn | w1, ..., wn-1) = count(w1 w2 ... wn) / count(w1 w2 ... wn-1) -- get these counts
			n_gram = self.train_tokens[i - (self.N - 1):i + 1]
			one_less = ' '.join(n_gram[:-1])
			n_gram = ' '.join(n_gram)
			
			# add to dict
			self.gram_count[n_gram] = 1 if n_gram not in self.gram_count else self.gram_count[n_gram] + 1
			self.gram_count[one_less] = 1 if one_less not in self.gram_count else self.gram_count[one_less] + 1
	
	def find_prob(self, n_gram, one_less):
		numerator = 0 if n_gram not in self.gram_count else self.gram_count[n_gram]
		denominator = 0 if one_less not in self.gram_count else self.gram_count[one_less]

		if self.smoothing == 'laplace':
			numerator += 1
			denominator += self.vocab_size

		elif self.smoothing == 'add-k':
			numerator += args.k
			denominator += args.k * self.vocab_size
			
		prob = numerator / denominator if denominator != 0 else 1e-10 # make it a very small value if it is 0

		return prob

	def compute_ppl(self):
		ppl = 0

		# find all N-gram and (N - 1)-gram counts based on our "training"
		# apply smoothing to outputs
		for i in range(self.N - 1, len(self.val_tokens)):
			n_gram = self.val_tokens[i - (self.N - 1):i + 1]
			one_less = ' '.join(n_gram[:-1])
			n_gram = ' '.join(n_gram)
			
			prob = self.find_prob(n_gram, one_less)
			ppl += math.log(prob)
		
		try:
			ppl = math.exp((-1 / self.N) * ppl)
		except OverflowError:
			ppl = float('inf')

		return ppl

	def to_display(self, tokens):
		# remove all SOS tokens, replace EOS tokens with .
		final = []
		for token in tokens:
			if token == '<s>':
				continue
			elif token == '</s>':
				if len(final) == 0:
					final.append('.')
				else:
					final[-1] += '.'
			else:
				final.append(token)
		return ' '.join(final)

	def generate(self, num_words):
		start = ['<s>'] * (self.N - 1)
		all_tokens = ['<s>'] * (self.N - 1)

		# this might be slow but it's the easiest way to do it
		for i in range(num_words):
			best, best_prob = '', 0
			one_less = ' '.join(start)

			# loop through the vocab and find the best word to put next, based on probability
			for word in self.vocab:
				tmp = start + [word]
				
				n_gram = ' '.join(tmp)
				prob = self.find_prob(n_gram, one_less)

				if prob > best_prob:
					best = word
					best_prob = prob
			
			all_tokens.append(best)

			start.append(best)
			start.pop(0) # remove first element; progress forward in n-gram

		return self.to_display(all_tokens)


ngram = NGram(args.corpus_path, int(args.N), int(args.vocab_size), args.smoothing)

print('ppl: ', ngram.compute_ppl())
print('generation: ', ngram.generate(args.num_words_generate))


