from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved models
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data)
        prediction = model.predict(vect)
        result = "SPAM" if prediction[0] == 0 else "HAM (Safe)"
        color = "#e74c3c" if prediction[0] == 0 else "#2ecc71"
        return render_template('index.html', prediction=result, color=color, text=message)

if __name__ == '__main__':
    app.run(debug=True)
