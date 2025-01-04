from flask import Flask, request, render_template
import joblib

tfidf=joblib.load("models/tfidfVectorizer.joblib")
rm_model=joblib.load("models/random_forest_model.pkl")

app=Flask(__name__)


model_metrics = [
    {"name": "Decision Tree", "accuracy": "96.6%", "precision": "96.0%", "recall": "95.7%", "f1": "95.8%"},
    {"name": "Random Forest", "accuracy": "98.3%", "precision": "98.2%", "recall": "97.6%", "f1": "97.8%"},
    {"name": "SVM", "accuracy": "98.3%", "precision": "98.1%", "recall": "97.5%", "f1": "97.7%"},
    {"name": "Naive Bayes", "accuracy": "96.7%", "precision": "96.3%", "recall": "96.0%", "f1": "96.1%"},
]

@app.route("/")
def index():
    return render_template("index.html", model_metrics=model_metrics)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "")
    message_tfidf = tfidf.transform([message]).toarray()
    prediction = rm_model.predict(message_tfidf)[0]
    probability = rm_model.predict_proba(message_tfidf)[0]  # Get probabilities
    spam_prob = round(probability[1] * 100, 2)  # Probability of Spam
    ham_prob = round(probability[0] * 100, 2) 
    if spam_prob > 40:
        label = "Spam"
    elif 20 <= spam_prob <= 40:
        label = "Might be Spam"
    else:
        label = "Ham"
    return render_template("index.html",model_metrics=model_metrics, prediction=label, input_message=message, spam_prob=spam_prob, ham_prob=ham_prob)


if __name__ == "__main__":
    app.run(debug=True)