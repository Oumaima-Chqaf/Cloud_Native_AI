import pickle

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model_gbm.pkl','rb'))

@app.route('/')
def index():
    return render_template('forms.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    # Convert the features to a NumPy array
    final_features = np.array(list(features.values())).reshape(1, -1)

    # Make predictions using your loaded SVM model
    prediction = model_gbm.best_estimator_.predict(final_features)

    # You can format the output as needed
    output = round(prediction[0], 2)

    return render_template('forms.html', prediction_text='Predicted glass type: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
