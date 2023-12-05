from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVC model
with open('SVC_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting values from the form
            pregnancies = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            blood_pressure = int(request.form['blood_pressure'])
            skin_thickness = int(request.form['skin_thickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetes_pedigree = float(request.form['diabetes_pedigree'])
            age = int(request.form['age'])

            # Creating a numpy array for prediction
            input_features = np.array(
                [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

            # Making prediction using the loaded model
            prediction = model.predict(input_features)

            # Returning the prediction to the HTML page
            return render_template('index.html', prediction_text='The prediction is {}'.format(prediction[0]))

        except ValueError as ve:
            return render_template('index.html', prediction_text='Please enter valid numerical values for all fields.')


if __name__ == '__main__':
    app.run(debug=True)
