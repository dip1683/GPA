#@author : Dipu
#date : 20Nov2020
#Project : GPA

import warnings as wr
wr.simplefilter(action='ignore')

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime as dtm
import json
import sys

app = Flask(__name__)

#app router
@app.route("/predict", methods=['POST'])

#Predict stockss
def predict():
    
    train_date = dtm.datetime(2020, 11, 18)
    json_ = request.json
    
    if json_['stockSymbol'] in {'SBIN','HDFCBANK', 'BLUEDART'} :
        try:
            
            d1 = dt.strptime(str(train_date), "%Y-%m-%d %H:%M:%S")
            d2 = dt.strptime(str(json_['date']), "%Y-%m-%d %H:%M:%S")
            days = abs((d2 - d1).days)
            
            if json_['stockSymbol'] =='SBIN':
                forecast = SBI.predict(n_periods = days)
            elif json_['stockSymbol'] =='BLUEDART':
                forecast = SBI.predict(n_periods = days)
            else :
                forecast = HDFC.predict(n_periods = days)
            
            response =  forecast[days-1]
            return jsonify({'requestDate':str(json_['date']),'stockPrice': str(response),'curSym': 'INR'})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        
        return ('No machine learning model for this stock found')
    
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 3000 # If you don't provide any port the port will be set to 12345

    SBI = joblib.load("sbi_arima.pkl") # Load "model.pkl"
    HDFC = joblib.load("hdfc_arima.pkl") # Load "model.pkl"
    BLUEDART = joblib.load("bluedart_arima.pkl") # Load "model.pkl"
    
    app.run(port=port, debug=True)