
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    news = data.get("news", "").strip()

    if not news:
        return jsonify({"error": "Please enter some news text."}), 400

    prediction = int(model.predict([news])[0])
    probabilities = model.predict_proba([news])[0]
    confidence = float(max(probabilities) * 100)

    label = "REAL" if prediction == 1 else "FAKE"

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
