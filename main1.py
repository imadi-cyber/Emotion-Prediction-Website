import sys
# coding=utf-8
import os
import glob
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import re
from nltk.corpus import stopwords
import tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
voc_size=10000
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from tensorflow.keras.preprocessing.sequence import pad_sequences
sent_length=35
from sklearn import preprocessing
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

df=pd.read_csv('Emotion_final.csv')
y=df['Emotion']

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)
y_final=np.array(y)
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


app = Flask(__name__)
model5 = load_model("weights1.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    stri = [str(x) for x in request.form.values()]
    str1=""
    for i in stri:
        str1 += i
    review = re.sub('[^a-zA-Z]', ' ', str1)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review,voc_size)] 
    embed = pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    predicti = model5.predict(embed)
    output = label_encoder.classes_[np.argmax(predicti)]

    return render_template('index.html', prediction_text='Text emotion would be - {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model5.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
