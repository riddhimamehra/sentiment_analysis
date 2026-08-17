import streamlit as st
import tensorflow as tf
import pickle

from text_preprocessing_utils import clean_text

model = tf.keras.models.load_model("sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


st.title("IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the Bidirectional LSTM model "
    "will predict whether the sentiment is positive or negative."
)

review = st.text_area(
    "Enter your movie review:",
    height=200,
    placeholder="Example: I really enjoyed this movie. The acting was amazing!"
)


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        cleaned_review = clean_text(review)

        sequence = tokenizer.texts_to_sequences([cleaned_review])

        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=600,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded_sequence, verbose=0)[0][0]

        predicted_class = 1 if prediction > 0.5 else 0

        predicted_sentiment = encoder.inverse_transform(
            [predicted_class]
        )[0]

        st.subheader("Prediction")

        if predicted_sentiment == "positive":
            st.success(f"Positive")
        else:
            st.error(f"Negative")

        st.write(f"Confidence: {prediction if predicted_class == 1 else 1 - prediction:.2%}")