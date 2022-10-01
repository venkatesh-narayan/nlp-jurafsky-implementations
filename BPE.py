import argparse
import re
import string
from nltk.tokenize import RegexpTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, help='path to corpus')
parser.add_argument('--k', type=int, help='number of merges to make')

args = parser.parse_args()

def tokenize(text):
	pattern = '\w+|\$[\d\.]+|\S+'
	tokenizer = RegexpTokenizer(pattern)
	
	tokens = tokenizer.tokenize(text)
	updated = []
	for token in tokens:
		if token not in string.punctuation:
			token = token.replace('\n', '_') # replace new lines with eos token (defining it as underscore)
			updated.append(token)
		else:
			if len(updated) == 0:
				continue
			else:
				updated[-1] = updated[-1] + '_' # add eos token to previous

	return updated

def get_word_dict_and_initial_vocab():
	with open(args.corpus_path, 'r') as f:
		text = f.read()
	
	tokens = tokenize(text) # get all tokens
	word_dict = dict() 		# keys are words, values are word counts
	initial_vocab = set()	# all unique characters used in the text file

	# always add EOS
	initial_vocab.add('_')

	for word in tokens:
		# add to dict if doesn't exist, otherwise increment
		spaced_word = ' '.join([char for char in word])
		
		word_dict[spaced_word] = (1 
								  if spaced_word not in word_dict 
								  else word_dict[spaced_word] + 1)
		
		# add all characters in each word to vocab
		for char in word:
			initial_vocab.add(char)
	
	return word_dict, initial_vocab

def BPE(k):
	word_dict, vocab = get_word_dict_and_initial_vocab()
	vocab = list(vocab) # convert to list to index through
	operations = [] # list combining operations that are done in BPE
	
	x = 0
	while x < k:
		best_pair = ('', -1, '', '') # merged string, count, first part, second part
		for i in range(len(vocab)):
			for j in range(i + 1, len(vocab)):
				# merge the current two entries in the vocab in both directions
				curr, curr_count = vocab[i] + ' ' + vocab[j], 0
				reverse, reverse_count = vocab[j] + ' ' + vocab[i], 0
				
				# get total counts for curr and reverse
				for key, val in word_dict.items():
					if curr in key:
						curr_count += val
					if reverse in key:
						reverse_count += val
				
				# update best pair if curr or reverse is more frequent than what it was before
				if curr_count > best_pair[1]:
					best_pair = (curr, curr_count, vocab[i], vocab[j])

				if reverse_count > best_pair[1]:
					best_pair = (reverse, reverse_count, vocab[j], vocab[i])

		# add the most frequent combination to the vocab and save the operation; break if it doesn't happen
		if best_pair[1] <= 0:
			break

		vocab.append(best_pair[2] + best_pair[3])
		operations.append((best_pair[2], best_pair[3]))

		# replace all instances of separate keys in word_dict with the combined
		old_word_dict = word_dict
		word_dict = {}
		for key, val in old_word_dict.items():
			check = best_pair[2] + ' ' + best_pair[3]
			if check in key:
				updated = key.replace(check, best_pair[2] + best_pair[3])
				word_dict[updated] = old_word_dict[key]
			else:
				word_dict[key] = val
		
		# save memory
		del old_word_dict

		x += 1
	
	return vocab, operations


def token_parser(text, operations):
	tokens = tokenize(text) # get all tokens in test text
	final_tokens = [] # tokens based on our vocabulary
	for token in tokens:
		spaced = ' '.join([char for char in token]) # add spaces to everything

		# greedily merge based on the operations we learned
		for first, second in operations:
			check = first + ' ' + second
			spaced = spaced.replace(check, first + second)
		
		# split on the spaces to get the final set of tokens for each token
		final_tokens.extend(spaced.split(' '))
	
	return final_tokens


vocab, operations = BPE(int(args.k))
print(vocab)

parsed_tokens = token_parser('lowerest worn', operations)
print(parsed_tokens)



