#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from flask import Flask,request,render_template
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
model = pickle.load(open('drugs_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    #int_features = [int_features]
    final_features = np.array(int_features)
    final_features = final_features.reshape(1, 5)
    prediction = model.predict(final_features)
    output = prediction[0]
    #print(output)
    return render_template('index.html',prediction_text = 'The Drug is : '+ output)
    
if __name__ == '__main__':
    app.run(port= 5000)


# In[ ]:




