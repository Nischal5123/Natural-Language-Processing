import operator
from collections import Counter
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np
import re  # regex
from nltk.corpus import stopwords


class UnimplementedFunctionError(Exception):
    pass


class Vocabulary:

    def __init__(self, corpus, threshold=0):
        #threshold param is to define the cut-off heuristic
        self.threshold = threshold
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)


    def most_common(self, k):
        freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, f in freq[:k]]

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

        # one possible tokenization is to ignore punctuation and capitalization then break on white-space then remove stopwords

        # Remove punctuation using regular expressions
        text = re.sub(r'[^\w\s]', '', text)
        # Convert text to lowercase
        text = text.lower()
        # Split text into words
        text = text.split()
        # Get a set of English stop words
        stop_words = set(stopwords.words('english'))
        # Filter out stop words
        filtered_words = [word for word in text if word not in stop_words]

        return filtered_words

    ###########################
    ## TASK 1.2            	 ##
    ###########################
    def build_vocab(self, corpus):
        """
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over


	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """

        #  -heuristric_threshold: to produce a vocab with or without thresholding heuristic


        word_count = {}
        for sentence in corpus:
            for word in self.tokenize(sentence):
                word_count[word] = word_count.get(word, 0) + 1

        filtered_words = [word for word in word_count if word_count[word] >= self.threshold]

        # Create the word2index and index2word mappings
        word2index = {word: i  for i, word in enumerate(filtered_words)}
        index2word = {i : word for i, word in enumerate(filtered_words)}
        freq = {word: word_count[word] for i, word in enumerate(filtered_words)}

        word2index['UNK'] = len(word2index)
        index2word[len(word2index)] = 'UNK'

        return word2index, index2word, freq

    ###########################
    ## TASK 1.3              ##
    ###########################
    def make_vocab_charts(self,cut_off):
        """
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    takes in cut_off heuristic

	    
	    """

        freqs = list(self.freq.values())

        # token id
        token_id = [self.word2idx[key] for key in self.freq.keys()]
        # token_id sorted based on frequency
        sorted_id_freq = sorted(zip(freqs, token_id), key=operator.itemgetter(0), reverse=True)
        sorted_freq, sorted_id = zip(*sorted_id_freq)
        cum_coverage = [sum(sorted_freq[:i + 1]) / sum(sorted_freq) for i in range(len(sorted_freq))]

        # Token Frequency
        fig1, ax = plt.subplots()
        ax.plot(range(len(sorted_id)), sorted_freq)
        ax.set_yscale('log')
        plt.axhline(y=cut_off, color='r', linestyle='-', label='freq=50')
        plt.xlabel('Token ID (sorted by frequency)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Token Frequency Distribution')
        plt.show()

        cutoff_index = next(idx for idx, value in enumerate(sorted_freq) if value <= cut_off)
        cutoff_cum_coverage = round(cum_coverage[cutoff_index],2)

        # Token Cumulative
        plt.plot(range(len(cum_coverage)), cum_coverage)
        plt.xlabel('Token ID (sorted by frequency)')
        plt.ylabel('Fraction of Token Occurences Covered')
        plt.title('Cumulative Frequency Covered')
        plt.axvline(x=cutoff_index, color='r', linestyle='-', label=cutoff_cum_coverage)
        plt.legend()
        plt.show()



