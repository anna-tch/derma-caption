import os
import os.path
import glob
import sys
import argparse
import pandas as pd
import numpy as np
from numpy import array
import nltk
from nltk.translate.bleu_score import corpus_bleu

import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau


# import modules
from modules.utils import *
from modules.extract_features import *
from modules.preprocess_captions import *
from modules.define_model import *
from modules.evaluate import *





def main():

	#parser = argparse.ArgumentParser()
	#parser.add_argument('--train', action='store_true')
	#parser.add_argument('--test', action='store_true')
	#args = parser.parse_args()

	print('\n\n\n#######################################################################')
	print('#  Load images ')
	print('#######################################################################\n\n\n')

	image_folder = "../images/"

	# get image ids
	images = glob.glob(image_folder + '*.jpg')
	num_images = len(images)
	print("All images : %d" % num_images)



	print('\n\n\n#######################################################################')
	print('#  Extract image features ')
	print('#######################################################################\n\n\n')


	feature_file = "../output/features.pkl"

	if os.path.isfile(feature_file):     # check if the file already exists
		print("Features have already been extracted and saved ...")
	else:
		features = extract_features(image_folder, num_images) # extract
		print("Extracted image features : %d" % len(features))
		dump_file(features, feature_file) # save
		print("Saved to a file...")



	print('\n\n\n#######################################################################')
	print('#  Preprocess captions ')
	print('#######################################################################\n\n\n')

	# read the doc
	doc = load_doc("original_data/captions.txt")

	# store to dictionary
	print("Loading descriptions...")
	descriptions = load_descriptions(doc)
	print(descriptions[next(iter(descriptions))])


	# clean the data
	print("\nCleaning descriptions...\n")
	descriptions_clean = clean_data(descriptions)
	print(descriptions_clean[next(iter(descriptions_clean))])

	# add tokens
	print("\nAdding start & end tokens...\n")
	descriptions_SE = add_se_tokens(descriptions_clean)
	print(descriptions_SE[next(iter(descriptions_SE))])




	print('\n\n\n#######################################################################')
	print('#  Split data into train / val / test sets ')
	print('#######################################################################\n\n\n')


	# train
	print("\nTrain :\n")
	train_image_id, train_descriptions, train_features = define_set(images, descriptions_SE, feature_file, train = True)

	# dev
	print("\nDev :\n")
	dev_image_id, dev_descriptions, dev_features = define_set(images, descriptions_SE, feature_file, dev= True)

	# test
	print("\nTest :\n")
	test_image_id, test_descriptions, test_features = define_set(images, descriptions_SE, feature_file, test= True)


	print('\n\n\n#######################################################################')
	print('#  Create embeddings')
	print('#######################################################################\n\n\n')

	# create a vocabulary
	print("\nCreating vocabulary...\n")
	vocab = create_reoccurring_vocab(train_descriptions, word_count_threshold = 5)
	print("Examples :")
	for word in vocab[:5]:
		print("=>",word)
	vocab_size = len(vocab) + 1
	print('Final vocab size : %d' % vocab_size)



	# calculate max length
	print("Calculate max length...")
	max_length = calculate_max_length(train_descriptions,85)
	print('Final max length of captions will be : %d'% max_length)



	# create indexes for words
	try:
		ixtoword, wordtoix = create_dict_of_indexes(vocab)
		print("\nWords are indexated...")
	except:
		print("\nWords are not indexated...")



	# load embd outside in order to make the model faster
	print("\nCreate an embedding layer...")
	embedding_layer = make_embedding_layer(vocab_size,
					wordtoix,
					embedding_dim=50,
					glove=True)



	print('\n\n\n#######################################################################')
	print('#  Train model ')
	print('#######################################################################\n\n\n')


	# define the model
	model = make_model(embedding_layer, max_length, vocab_size)

	# compile
	model.compile(loss=masked_loss_function, optimizer= 'adam')

	ep = 1
	epochs = 270
	batch_size= 32
	steps = len(train_descriptions)//batch_size
	history={'loss':[], 'BLEU_val':[]}
	Reduce_lr=ReduceLROnPlateau(monitor='loss',
						factor=0.9,
						patience=5,
						verbose=0,
						mode='auto',
						min_delta=0.0001,
						min_lr=0.000001)

	for i in range(ep,epochs):
		print('\nEpoch :',i,'\n')

		# create the data generator
		generator = data_generator(train_descriptions,
					train_features,
					wordtoix,
					max_length)
		# fit for one epoch
		h = model.fit_generator(generator,
				epochs=1,
				steps_per_epoch=steps,
				verbose=1,
				callbacks=[Reduce_lr] )

		#model.save('../output/model_' + str(i) + '.h5')
		ep = i + 1
		history['loss'].append(h.history['loss'])


		# save model every 10 epochs
		if i % 3 == 0:
			print("\nSave the model...\n")
			model.save('../output/model_' + str(i) + '.h5')
			bleu_eval= evaluate_model(model,
							dev_descriptions,
							dev_features,
							wordtoix,
							ixtoword,
							max_length,
							K_beams=1)

			history['BLEU_val'].append((bleu_eval,i))

	print('Finished!')


	print('\n\n\n#######################################################################')
	print('#  Test model ')
	print('#######################################################################\n\n\n')

	# load the model
	#list_of_files = glob.glob('../output/*.h5')
	#latest_file = max(list_of_files, key=os.path.getctime)
	#print("Loading the latest model : {}".format(latest_file))
	#latest_model = keras.models.load_model(latest_file, custom_objects={'masked_loss_function': masked_loss_function})

	# evaluate with bleu score, store results
	actual, predicted, image_ids = evaluate_model(model,
					test_descriptions,
					test_features,
					wordtoix,
					ixtoword,
					max_length)


	# normalize the results
	norm_actual, norm_predicted = normalize_ref_and_pred(actual, predicted)


	# save
	df = pd.DataFrame.from_dict(norm_actual, orient='index', columns=['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5'])
	df.insert(0, "image_id", image_ids, True)
	df.insert(1, "predicted", norm_predicted, True)
	df.to_csv('../results.csv', encoding='utf-8', sep = '\t')





	return True


if __name__ == '__main__':
	main()
