import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)
price_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    # Render the template with an empty form
    return render_template('index.html', prediction_text='', City='', type='', room_number='', Area='', Street='', city_area='', condition='', furniture='', description='', hasBalcony='')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    City = request.form.get('City')
    type = request.form.get('type')
    room_number = int(request.form.get('room_number'))
    Area = int(request.form.get('Area'))
    Street = request.form.get('Street')
    city_area = request.form.get('city_area')
    condition = request.form.get('condition ')
    furniture = request.form.get('furniture ')
    description = request.form.get('description ')
    hasBalcony = bool(request.form.get('hasBalcony '))

    # Create a dictionary from the form data
    data = {
        'City': [City],
        'type': [type],
        'room_number': [room_number],
        'Area': [Area],
        'Street': [Street],
        'city_area': [city_area],
        'condition ': [condition],
        'furniture ': [furniture],
        'description ': [description],
        'hasBalcony ': [hasBalcony ]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data, index=[0])

    # Pass the DataFrame to your API function
    prediction = price_model.predict(df)[0]

    output_text = "The Asset's price is: {}".format(prediction)

    # Render the template with the prediction text and an empty form
    return render_template('index.html', prediction_text=output_text, City='', type='', room_number='', Area='', Street='', city_area='', condition='', furniture='', description='', hasBalcony='')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

