from flask import Flask , request
import numpy as np
import keras
from keras.models import load_model
import joblib
import flask

print(flask.__version__)

app = Flask(__name__)
model = load_model("./model_inception.hdf5")
scaler = joblib.load("./scaler.save")

# python -m flask --app index run

@app.route("/predict", methods=['POST'])
def predict():
    # recive data from the user
    request_data = request.get_json()
    
    can_id = int(request_data['can_id'], 16)  # Convert hex to int
    dlc = int(request_data['dlc'])
    data = [int(request_data['data'][i], 16) for i in range(8)]  # Convert hex to int

    # Preprocess the input data as necessary
    input_data = np.array([[can_id, dlc] + data ])

    input_data = scaler.transform(input_data)


    print(input_data)
    ## test data
    # can_id = -1.58416054
    # dlc = 0
    # d0=-0.67226711
    # d1= -0.90624512
    # d2=-0.82496983
    # d3=-0.99551724
    # d4=-0.83802093
    # d5=-0.71949151
    # d6=-0.44731662
    # d7=-0.73740763
    # Preprocess the input data as necessary

    prediction = model.predict(input_data)
    return str(prediction[0][0])