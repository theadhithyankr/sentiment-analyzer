from flask import Flask, render_template, request
import joblib
import time
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load trained models and vectorizer
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
nb_model = joblib.load(MODEL_DIR / "nb_model.pkl")
svm_model = joblib.load(MODEL_DIR / "svm_model.pkl")

# Sentiment score mapping (0 = Negative, 50 = Neutral, 100 = Positive)
sentiment_mapping = {
    'negative': 0,
    'mixed': 33,
    'neutral': 50,
    'positive': 100
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("text", "").strip()
        if not user_input:
            return render_template("index.html", user_input="")

        X_input = vectorizer.transform([user_input])

        # Measure processing time
        start_nb = time.time()
        nb_prediction = nb_model.predict(X_input)[0]
        end_nb = time.time()

        start_svm = time.time()
        svm_prediction = svm_model.predict(X_input)[0]
        end_svm = time.time()

        nb_time = round(end_nb - start_nb, 5)
        svm_time = round(end_svm - start_svm, 5)

        # Calculate sentiment score for scale
        sentiment_score = sentiment_mapping.get(nb_prediction, 50)  # Default to neutral

        # Fake accuracy metrics (replace with real ones if available)
        nb_accuracy, nb_precision, nb_recall, nb_f1 = 0.80, 0.78, 0.76, 0.77
        svm_accuracy, svm_precision, svm_recall, svm_f1 = 0.77, 0.75, 0.73, 0.74

        return render_template(
            "index.html",
            user_input=user_input,
            nb_result=nb_prediction,
            svm_result=svm_prediction,
            nb_accuracy=nb_accuracy,
            nb_precision=nb_precision,
            nb_recall=nb_recall,
            nb_f1=nb_f1,
            nb_time=nb_time,
            svm_accuracy=svm_accuracy,
            svm_precision=svm_precision,
            svm_recall=svm_recall,
            svm_f1=svm_f1,
            svm_time=svm_time,
            sentiment_score=sentiment_score  # Sentiment pointer position
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
