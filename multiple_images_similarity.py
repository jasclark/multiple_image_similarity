import sqlite3, pickle, math, os, numpy as np
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy


#initilize app and db
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sim.db'
db = SQLAlchemy(app)
port = int(os.getenv('PORT', 5000))

class Image(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	url = db.Column(db.Text)
	vector = db.Column(db.PickleType)

	def __init__(self, url, vector):
	    self.url = url
	    self.vector = vector

@app.route('/')
def show_images():
	images = Image.query.all()
	return render_template('show_images.html', images=images)

@app.route('/show_similar_images')
def show_similar_images():
	target_image_ids = map(int, request.args.get('image_ids').split(','))
	target_images = Image.query.filter(Image.id.in_(target_image_ids)).all()	

	#add target_images vectors together
	length = len(target_images[0].vector)
	target_vector = np.zeros(length)
	for target_image in target_images:
		target_vector = np.add(target_vector, target_image.vector)

	#normalize
	target_vector_length = math.sqrt(np.sum(np.square(target_vector)))
	target_vector = np.divide(target_vector, target_vector_length)

	#create matrix out of all vectors
	image_list = []
	images = Image.query.all()
	for image in images:
		image_list.append(image.vector)
	matrix = np.asarray(image_list)

	#calculate cosine similarity
	cosine_similarities = linear_kernel(target_vector, matrix)[0]	
	similar_indices = cosine_similarities.argsort()[::-1].tolist()
	similar_indices = [x+1 for x in similar_indices]

	#sort images by similarity score
	similar_images = Image.query.filter(Image.id.in_(similar_indices)).all()
	sorted_similar_images = []
	for similar_index in similar_indices:
		for similar_image in similar_images:
			if similar_index == similar_image.id:

				#subtract one from similar_image.id because python lists are 0 indexed
				similar_image.score = cosine_similarities[similar_image.id-1]
				sorted_similar_images.append(similar_image)

	return render_template('show_similar_images.html', images=sorted_similar_images,
		target_ids=target_image_ids, target_images=target_images)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=port, debug=True)