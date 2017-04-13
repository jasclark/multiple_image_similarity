import pickle, json, math, numpy as np
from multiple_images_similarity import Image, db
from watson_developer_cloud import VisualRecognitionV3

#iterate through classifiers and append each class to a list
def create_class_list(response):
	classifiers = response["images"][0]["classifiers"]
	classes = []
	for classifier in classifiers:
		for c in classifier['classes']:
			classes.append((c["class"], c["score"]))
	return classes

#create database and load urls from file
db.create_all()
with open('image_urls', 'rb') as fp:
	urls = pickle.load(fp)

#classify urls
visual_recognition = VisualRecognitionV3('2016-05-20', api_key='removed for github')
image_classes = []
responses = []
for idx, url in enumerate(urls):
	response = visual_recognition.classify(images_url=url)
	responses.append(response)
	classes = create_class_list(response)
	image_classes.append(classes)

#save responses to disk as backup in case db fails
with open('responses', 'wb') as fp:
	pickle.dump(responses, fp)

#create master list of classes
all_classes = []
for image in image_classes:
	for image_class in image:

		#grab first part of class tuple created in create_class_list()
		image_class = image_class[0]
		if image_class not in all_classes:
			all_classes.append(image_class)

#create vector for each image where indices of vector correspond with all_classes
#and where the value of each class is the score provided by the classifiers
for idx, image in enumerate(image_classes):
	image_vector = np.zeros(len(all_classes))
	for image_class in image:
		index = all_classes.index(image_class[0])
		image_vector[index] = image_class[1]

	#normalize vector
	vector_length = math.sqrt(np.sum(np.square(image_vector)))
	image_vector = np.divide(image_vector, vector_length)
	
	#add each image url and vector to the database
	image = Image(urls[idx], image_vector)
	db.session.add(image)

#commit to db
db.session.commit()

