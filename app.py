from flask import Flask, render_template, request

import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_path = r'C:\Users\ACER\Documents\ml prjt\amazon review\lstm_model.h5'
tokenizer_path = r"C:\Users\ACER\Documents\ml prjt\amazon review\tokenizer .pkl"

model = load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Helper function to clean text
def clean_text(text):
    return text.lower().strip()

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer, max_sequence_length=100):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😡"
    return sentiment, float(prediction)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for processing the form input and returning the prediction
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
    return render_template('index.html', prediction=sentiment, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
