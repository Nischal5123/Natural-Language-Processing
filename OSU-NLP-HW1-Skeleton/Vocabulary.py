from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np


class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """

		tokenized=text.lower().split()

		return tokenized
		# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		#raise UnimplementedFunctionError("You have not yet implemented tokenize.")



	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self,corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """
		heuristric_threshold=10
		word_count = {}
		for sentence in corpus:
			for word in self.tokenize(sentence):
				word_count[word] = word_count.get(word, 0) + 1
		filtered_words=[word for word in word_count if word_count[word] >= heuristric_threshold]

		# Create the word2index and index2word mappings
		word2index = {word: i + 1 for i, word in enumerate(filtered_words)}
		index2word = {i + 1: word for i, word in enumerate(filtered_words)}
		freq= {word: word_count[word] for i, word in enumerate(filtered_words)}

		# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		#raise UnimplementedFunctionError("You have not yet implemented build_vocab.")
		return word2index,index2word,freq


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """
		freqs = list(self.freq.values())
		cum_coverage = [sum(freqs[:i + 1]) / sum(freqs) for i in range(len(freqs))]

		# Plot word frequency chart
		plt.bar(range(len(freqs)), np.log(freqs))
		plt.xlabel('Word Index')
		plt.ylabel('Frequency')
		plt.title('Word Frequency Chart')
		plt.show()

		# Plot cumulative coverage chart
		plt.plot(range(len(cum_coverage)), cum_coverage)
		plt.xlabel('Vocabulary Size')
		plt.ylabel('Cumulative Coverage')
		plt.title('Cumulative Coverage Chart')
		plt.show()
	    # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		#raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")

