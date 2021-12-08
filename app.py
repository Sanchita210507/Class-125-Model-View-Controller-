from flask import Flask, jsonify, request
from classifier import get_prediction

app = Flask(__name__)

@app.route("/")
def intro():
    return "Welcome to the homepage!"

@app.route("/predict-digit", methods = ["POST"])
def predict_data():
    img = request.files.get("digit")
    pred = get_prediction(img)
    return jsonify({
        "prediction": pred
    })

if __name__ == "__main__":
    app.run(debug = True, port = 8000)