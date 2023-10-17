import pickle

from flask import Flask
from flask import request
from flask import jsonify


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('dv.bin')
model = load('model1.bin')

app = Flask('credit_probability')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        client_data = request.json
        print("Received client data:", client_data)
        X_client = dv.transform([client_data])
        print("Transformed data:", X_client)
        probs = model.predict_proba(X_client)
        probability = probs[0][1]
        print("Probability for class 1:", probability)
        result = {'get_credit_probability': float(probability)}
        return jsonify(result)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=9696)