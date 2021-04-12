from flask import Flask, render_template,request
import pickle as pkl
import numpy as np

with open('model.pkl','rb') as f:
    model = pkl.load(f)

app = Flask(__name__)    
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    inf1 = request.form['1']
    inf2 = request.form['2']
    inf3 = request.form['3']
    inf4 = request.form['4']
    arr = np.array([[inf1,inf2,inf3,inf4]])
    output = model.predict(arr)
    return render_template('predict.html',data=output)

if __name__ == '__main__':
    app.run(debug=False)