from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/iris',methods=['POST'])
def predict():
   features=[float(x) for x in request.form.values()] 
   features=[np.array(features)] 
   output=model.predict(features)
   output=output[0]
   return render_template ('result.html',prediction_text="The Flower is - {}".format(output))
if __name__=='__main__':
    app.run(debug=True)

