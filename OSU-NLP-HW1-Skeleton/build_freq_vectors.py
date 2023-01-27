import os
from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE


import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
	"""

	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns:
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

	    """
	N = vocab.size
	window_size = 4
	file_name = 'C_Matrix.npy'
	if os.path.isfile(file_name):
		C = np.load(file_name)
		logging.info("Loading Matrix from Disk")
	else:
		logging.info("Creating Matrix from Scratch")
		C = np.zeros((N, N))
		for sentence in tqdm(corpus):
			tokens = vocab.text2idx(sentence)
			tokens_len = len(tokens)
			for i in range(tokens_len):
				# Given a word at position i in some text, the k window of that wod is all other words in the range i-k to i+k
				main_token=tokens[i]
				for j in range(max(i - window_size, 0), min(i + window_size + 1, tokens_len)):
						C[main_token, tokens[j]] += 1
		np.save('C_Matrix.npy', C)
	return C

	

###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	"""
	    
	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	    """

	C = compute_cooccurrence_matrix(corpus, vocab)
	# The small constant added to C before computing PPMI is to avoid the term in the log (C_ij N / C_ii C_jj) from being log(0) when words i and j do not co-occur
	N = vocab.size
	epsilon = 1e-5
	C = C + epsilon
	row_sum = C.sum(axis=1)
	col_sum = C.sum(axis=0)
	total_sum = C.sum()
	PPMI = np.zeros((N, N))

	for i in tqdm(range(N)):
		for j in range(N):
			PPMI[i, j] = max(np.log(C[i, j] * total_sum / row_sum[i] * col_sum[j]), 0)
	return PPMI


################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]


	logging.info("Building vocabulary")
	#cut_off heuristic to test
	cut_off=50

	#explore vocab without any cut-off heuristic
	vocab_test = Vocabulary(dataset_text)
	logging.info("Creating Plots")
	vocab_test.make_vocab_charts(cut_off)
	plt.close()
	plt.pause(0.01)

	#vocab with best threshold identified from the plots
	vocab = Vocabulary(dataset_text,cut_off)



	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)
	plt.savefig('tsne.pdf')
	plt.show()

def plot_loss(loss):
	"""

	plots avg. loss over last 100 batches


	"""
	fig1, ax = plt.subplots()
	ax.plot(range(len(loss)), loss)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Average Epoch Loss')
	plt.savefig('loss.pdf')
	plt.show()


if __name__ == "__main__":
    main_freq()

