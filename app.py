import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("ECG-1.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            4
            
            print("prediction",preds)
            
        index = ['Fusion Beat','Normal Beat','Unknown Beat','Supraventricular ectopic Beat','Ventricular ectopic beat']
        
        
        text = "The predicted type is " + str(index[preds[0]]) +".To know more about this scroll down and check with the ECG class description."      
   
    return text
if __name__ == '__main__':
    app.run(debug = True)
        
        
        
    
    
    