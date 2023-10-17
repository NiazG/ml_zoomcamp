from flask import Flask, request, jsonify
import pickle

with open('model2.bin', 'rb') as f:
    model = pickle.load(f)
with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

app = Flask('q6')

@app.route('/predict', methods=['POST'])
def q6_question():
    client = request.get_json()

    X = dv.transform(client)
    y_pred = model.predict_proba(X)[0,1]
    cred = y_pred>= 0.5

    result = {
        "Probability": round(float(y_pred), 3),
        "Good Credit": bool(cred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=9696)