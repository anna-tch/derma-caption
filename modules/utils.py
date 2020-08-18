import pickle
import sys

from pickle import load

def progressBar(value, endvalue, bar_length=20,job='Job'):
	''' shows progress of the process
	INPUT :
	OUTPUT :
	'''

	# get values
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))

	# update the bar
	sys.stdout.write("\r{0} Completion: [{1}] {2}%".format(job,arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()


def dump_file(fic, folder):
	''' saves dictionary to pickle file
	INPUT : dict
	OUTPUT : True if file is created
	'''
	pickle.dump(fic, open(folder, 'wb'))
	return True


def split_descriptions(descriptions, dataset):
	'''
	INPUT: descriptions(dict), dataset (list)
	OUTPUT: dataset (dict)
	'''
	new_dataset = {}
	for image_id in dataset:
		new_dataset[image_id] = descriptions[image_id]
	return new_dataset


def load_photo_features(dataset, feature_file = '../output/features.pkl'):
	''' opens a pkl document, and takes the features
	that correspond to the given dataset
	INPUT : document (pkl), dataset (list)
	OUTPUT : features (dict)
	'''
	# load all features
	all_features = pickle.load(open(feature_file, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features


def define_set(images, descriptions, features, train=False, dev=False, test=False):
	"""
	define datasets 80-10-10
	INPUT :
	OUTPUT :
	"""

	# define the set
	split_1 = int(0.8 * len(images))
	split_2 = int(0.9 * len(images))

	if train:
		image_set = images[:split_1]
	if dev:
		image_set = images[split_1:split_2]
	if test:
		image_set = images[split_2:]



	# load photo features
	image_id = [line.split('.jpg')[0].split('/')[-1] for line in image_set]
	print('Ids : %d' % len(image_id))
	#print(image_id)

	# load photo features
	descriptions_set = split_descriptions(descriptions, image_id)
	print('Descriptions : %d' % len(descriptions_set))

	# load photo features
	features_set = load_photo_features(image_id, features)
	print('Images : %d' % len(features_set))

	return  image_id, descriptions_set, features_set
