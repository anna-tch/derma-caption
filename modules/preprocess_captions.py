import string
import os
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.layers import Embedding


from modules.utils import progressBar


def load_doc(filename):
	''' reads text file
	INPUT : document (txt)
	OUTPUT : text (str)
	'''
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


def load_descriptions(doc):
	'''
	INPUT : descriptions (str)
	OUTPUT : descriptions (dict)
	'''
	descriptions = {}
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) > 1:
			# get image id and description
			tokens = line.split()
			image_id, image_desc = tokens[0], tokens[1:]
			# save image id without extension
			image_id = image_id.split('.')[0]
			# convert description tokens back to string
			image_desc = ' '.join(image_desc)
			if image_id not in descriptions:
				descriptions[image_id] = list()
			descriptions[image_id].append(image_desc)
	return descriptions


def clean_data(descriptions):
	'''tokenizes the text data, converts to lower case,
	removes punctuation, indefinite articles, numbers
	INPUT: descriptions (dict)
	OUTPUT: clean descriptions (dict)
	'''
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	# get a list of articles, it's possible to add the article
	# 'the' too, if needed
	#articles = ['a', 'an']
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove hanging 'an'
			#desc = [word for word in desc if word not in articles]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

	print('Data cleaned : %d' % len(descriptions))
	return descriptions


def add_se_tokens(descriptions, start_token = '<startseq>', end_token = '<endseq>'):
	''' adds start and end tokens
	INPUT: descriptions (dict)
	OUTPUT: descriptions (dict) with start and end tokens
	'''
	# interate over all image ids
	for key in descriptions:
		# iterate over all five descriptions under this id
		for i in range(len(descriptions[key])):
			# add start and end tokens
			descriptions[key][i] = start_token + ' ' + descriptions[key][i] + ' ' + end_token
	return descriptions


def create_list_of_values(descriptions):
	'''interate over all values and save them to a list
	INPUT : descriptions (dict) where one key corresponds to a list of values
	OUTPUT : descriptions (list)
	'''
	# Create a list of all the captions
	all_descriptions = []
	for key, val in descriptions.items():
		for desc in val:
			all_descriptions.append(desc)
	return all_descriptions


def create_reoccurring_vocab(descriptions, word_count_threshold = 10):
	'''making a vocabulary of the words that occur
	more than word_count_threshold time
	INPUT : descriptions (dict)
	OUTPUT : vocab (dict)
	'''
	# Create a list of all the captions
	# Create a list of all the captions
	all_captions = []
	for key, val in descriptions.items():
		for cap in val:
			all_captions.append(cap)

	# Consider only words which occur at least 10 times in the corpus
	word_counts = {}
	nsents = 0
	for sent in all_captions:
		nsents += 1
		for w in sent.split(' '):
			word_counts[w] = word_counts.get(w, 0) + 1

	vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

	print('Vocab Size %d ' % len(vocab))
	return vocab


def calculate_max_length(desc,p):
	all_desc = []
	# Create a list of all the captions
	for i in desc:
		for j in desc[i]:
			all_desc.append(j)

	# finding the maximum length of questions and answers
	# because there are senteces with unusually long lengths,
	# we caculate the max length that p% of data can be placed in
	length_all_desc = list(len(d.split()) for d in all_desc)

	print('Percentile {} of len of questions: {}'.format(p,np.percentile(length_all_desc, p)))
	print('Longest sentence: ', max(length_all_desc))

	return int(np.percentile(length_all_desc, p))


def trimRareWords(desc):
	# only keep the decriptions that have the words from our vocab
	num_des = 0
	num_trim = 0
	desc_result = desc.copy()
	for d in desc_result:
		desc_result[d]=[]

	# Filter out pairs with trimmed words
	i=0
	for d in desc:
		i+=1
		progressBar(value=i, endvalue=len(desc))
		for p in desc[d]:
			num_des += 1
			keep_input = True
			# Check input sentence
			for word in p.split(' '):
				if word not in vocab:
					keep_input = False
					break

			# Only keep descriptions that do not contain trimmed word(s) in them
			if keep_input:
				num_trim += 1
				desc_result[d].append(p)

	print("\nTrimmed from {} pairs to {}".format(num_des, num_trim))
	return desc_result


def create_dict_of_indexes(vocab):
	ixtoword = {} # index to word dic
	wordtoix = {} # word to index dic

	ix = 1
	for w in vocab:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	return ixtoword, wordtoix


def load_glove(embedding_dim):

	embeddings_index = {}
	glove_dir = '../embeddings/'
	f = open(os.path.join(glove_dir + 'glove.6B.'+str(embedding_dim)+'d.txt'), encoding="utf-8")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def make_embedding_layer(vocab_size, wordtoix, embedding_dim=50, glove=True):
	if glove == False:
		embedding_matrix = np.zeros((vocab_size, embedding_dim))
		print('Just a zero matrix loaded')
	else:
		embeddings_index = load_glove(embedding_dim)
		print('GloVe loaded!')

		# to import as weights for Keras Embedding layer
		embedding_matrix = np.zeros((vocab_size, embedding_dim))
		for word, i in wordtoix.items():
			# Get x-dim dense vector for each of the vocab_rocc
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# Words not found in the embedding index will be all zeros
				embedding_matrix[i] = embedding_vector

	# create an embedding layer
	embedding_layer = Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=False)
	embedding_layer.build((None,))
	embedding_layer.set_weights([embedding_matrix])

	return embedding_layer
