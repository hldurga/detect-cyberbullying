from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        user_input = request.form['text']
        try:
            transformed_input = vectorizer.fit_transform([user_input])  # Use transform, not fit_transform
            prediction = model.predict(transformed_input)[0]
        except Exception as e:
            error = f"Error during prediction: {str(e)}"
    
    return render_template('index.html', prediction=prediction, error=error)

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
