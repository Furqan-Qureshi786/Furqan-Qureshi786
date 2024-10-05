import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda text: hub_layer(text), input_shape=[], dtype=tf.string))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Uncomment if you have a pre-trained model
    # model.load_weights('path_to_saved_model_weights')

    return model

model = load_model()

# Load and preprocess your dataset
try:
    train_data = pd.read_csv('https://raw.githubusercontent.com/Furqan-Qureshi786/TextClassifier/refs/heads/main/wine-reviews.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Check if the necessary columns exist
if 'description' not in train_data.columns or 'points' not in train_data.columns:
    st.error("The required columns 'description' or 'points' are not present in the dataset.")
else:
    # Convert 'points' to numeric, handling errors by coercing to NaN
    train_data['points'] = pd.to_numeric(train_data['points'], errors='coerce')
    train_data.fillna({'points': 0}, inplace=True)  # Fill NaN values for points column

    # Prepare the data for training (features and labels)
    X = train_data['description']  # Feature column
    y = train_data['points']  # Target variable; ensure this is numeric

    # Streamlit app interface
    st.title('Wine Review Quality Classifier')

    # Text input for wine review
    review_input = st.text_area('Enter your wine review:', '')

    # Predict button
    if st.button('Predict'):
        if review_input.strip() == '':
            st.error('Please enter a wine review.')
        else:
            try:
                # Ensure the input is in the correct format (a list of strings)
                review_array = tf.convert_to_tensor([review_input], dtype=tf.string)  # Convert directly to tensor
                pred_prob = model.predict(review_array)[0][0]

                # Determine label based on the predicted probability
                label = 'High Quality' if pred_prob >= 0.5 else 'Low Quality'

                # Show result
                st.subheader(f'Prediction: {label}')
                st.text(f'Probability of being High Quality: {pred_prob:.2f}')
            except Exception as e:
                st.error(f"Error during prediction: {e}")
