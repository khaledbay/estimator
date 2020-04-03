#from email.mime.text import MIMEText
from flask import Flask, render_template, request, url_for, flash
#from flask_mail import Mail, Message
import pandas as pd
import pickle
#import smtplib
#import smtplib

def create_app():
  app = Flask(__name__)
  #Bootstrap(app)

  return app

app = create_app()
#app = Flask(__name__)
#mail=Mail(app)
#app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Setting secret key, if you dont while flashing if will show exception.

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/')
@app.route('/hello_world', methods=["GET", "POST"])
def hello_world():
    if(request.method == "POST"):
        myDict = request.form

        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        breathDiff = int(myDict['breathDiff'])

        # Input Features
        inputFeatures = [fever, age, bodyPain, runnyNose, breathDiff]

        # Predicting new Values 
        infProb = clf.predict_proba([inputFeatures])[0][1]

        return render_template('show.html', inf=round(infProb * 100))
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
    
