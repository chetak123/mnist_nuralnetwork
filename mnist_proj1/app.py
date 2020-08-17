import pickle
import os
#import libraries
import numpy as np
import scipy.special #here its used for the activation function(writting clear sigmoid fun.)
import matplotlib.pyplot as plt
def nural(input_l):
    x_act=lambda x:scipy.special.expit(x)
    inp=np.array(input_l,ndmin=2).T
    hi_i=np.dot(x_wih,input_l)
    hi_o=x_act(hi_i)
    fi_i=np.dot(x_who,hi_o)
    fi_o=x_act(fi_i)
    return (fi_o)
model_path1=os.path.join(os.path.pardir,'models','model_reg.pkl') #path variable where file is stored
model_path2=os.path.join(os.path.pardir,'models','model_who.pkl') #path variable where file is stored
model_path3=os.path.join(os.path.pardir,'models','model_wih.pkl') #path variable where file is stored
#checking the saved model through already available path variable model_path
model_pick1=open(model_path1,'rb') #rb means read only in binary format (since its saved in binary not UTF8)
model_pick2=open(model_path2,'rb') #rb means read only in bi
model_pick3=open(model_path3,'rb') #rb means read only in bi
model_load1=pickle.load(model_pick1)
x_wih=pickle.load(model_pick3)
x_who=pickle.load(model_pick2)
model_pick1.close()
model_pick2.close()
model_pick3.close()


#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
#import keras.models
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
import base64
#tell our app where our saved model is
#sys.path.append(os.path.abspath("./model"))
#from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
#global model, graph
#initialize these variables
#model, graph = init()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		#img_by=bytes(imgstr,)
		output.write(base64.b64decode(imgstr))

	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	print(imgData)
	#encode it into a suitable format
	#imgData=imgData.decode()
	convertImage(imgData)
	print ("debug")
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	#x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	#x = x.reshape(1,28,28,1)
	print ("debug2")
	print (type(x))
	print (np.shape(x))
	#img_data  = x.reshape(784)
    # then scale data to range from 0.01 to 1.0
	#img_data = ((img_data / 255.0 * 0.99) + 0.01)
	label=0

	img_data  = 255.0 - x.reshape(784)
	print(img_data)

	print (np.shape(img_data))
    # then scale data to range from 0.01 to 1.0
	img_data = (img_data / 255.0 * 0.99) + 0.01
	#inputs = (np.asfarray(img_data) / 255.0 * 0.99) + 0.01
	#outputs = model_load1(inputs)
	#label=model_load1((np.asfarray(img_data)/255.0*0.99)+0.01)
	label=model_load1(img_data)
	#print(outputs)
    # the index of the highest value corresponds to the label
	label = np.argmax(label)
	print(label)
	label=label.tolist()
	label=str(label)

	#in our computation graph
	#with graph.as_default():
		#perform the prediction
		#"""out = model.predict(x)
		#						print(out)
		#						print(np.argmax(out,axis=1))"""
	print ("debug3")
		#convert the response to a string
		#response = np.array_str(np.argmax(out,axis=1))
	response="photo"
	return (label)	
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)