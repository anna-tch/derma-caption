import os

# classifier
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input


# progressBar
from modules.utils import progressBar

def load_resnet_classifier():
	"""
	downloads and restructures ResNet50 model
	INPUT :
	OUTPUT : model (without the last two layers)
	"""
	# load the model
	model = ResNet50()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize
	print(model.summary())

	return model


def extract_features(directory, num_images):
	"""
	resizes, reshapes, preprocess images with a classifier
	creates a dictionary of the extracted features
	INPUT : directory
	OUTPUT : dict (features)
	"""
	# load the model
	model = load_resnet_classifier()
	# extract features from each photo
	features = dict()
	i = 0
	for name in os.listdir(directory):
		# consider only images from the folder
		if ".jpg" in name:
			progressBar(value=i,endvalue=num_images)
			i +=1
			# load an image from file
			filename = directory + name
			image = load_img(filename, target_size=(224, 224))
			# convert the image pixels to a numpy array
			image = img_to_array(image)
			# reshape data for the model
			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			# prepare the image for the resnet model
			image = preprocess_input(image)
			# get features
			feature = model.predict(image, verbose=0)
			# get image id
			image_id = name.split('.')[0]
			# store feature
			features[image_id] = feature.reshape(2048)
			#print('>%s' % name)
		else:
			continue

	return features
