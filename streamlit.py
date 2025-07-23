import streamlit as st
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model_path = r'C:\Users\ACER\Documents\ml prjt\amazon review\lstm_model.h5'
tokenizer_path = r"C:\Users\ACER\Documents\ml prjt\amazon review\tokenizer .pkl"


model = load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)


def clean_text(text):
    return text.lower().strip()


def predict_sentiment(text, model, tokenizer, max_sequence_length=100):
    
    text = clean_text(text)
   
    sequence = tokenizer.texts_to_sequences([text])
    
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    
    prediction = model.predict(padded_sequence)[0][0]
    
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😡"
    
    return sentiment, float(prediction)

st.title("Sentiment Analysis Web App")
st.markdown("Enter a sentence below, and the model will predict if the sentiment is positive or negative.")

user_input = st.text_area("Enter text here")

if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
