from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("student_model.pkl", "rb"))

@app.route("/")
def home():
    return "Student Score Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return jsonify({"predicted_average_score": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
