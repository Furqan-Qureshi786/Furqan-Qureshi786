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
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    # Load trained weights (this assumes you have trained and saved model weights)
    # Example: model.load_weights('path_to_saved_model_weights')
    return model

model = load_model()

# Load and preprocess your dataset
train_data = pd.read_csv('https://github.com/Furqan-Qureshi786/TextClassifier/blob/main/wine-reviews.csv')

# Convert specific columns to numeric, handling errors by coercing to NaN
columns_to_convert = ['points', 'price']  # Only numeric columns
for column in columns_to_convert:
    train_data[column] = pd.to_numeric(train_data[column], errors='coerce')

# Fill NaN values after conversion
train_data.fillna(0, inplace=True)  # You can also use train_data.dropna(inplace=True)

# Prepare the data for training (split features and labels)
# Assuming 'quality' is your target variable; adjust as needed
X = train_data['description']  # Feature column
y = train_data['quality']  # Target variable

# Optionally: Split your data into training and validation sets here

# Streamlit app interface
st.title('Wine Review Quality Classifier')

# Text input for wine review
review_input = st.text_area('Enter your wine review:', '')

# Predict button
if st.button('Predict'):
    if review_input.strip() == '':
        st.error('Please enter a wine review.')
    else:
        # Prediction
        pred_prob = model.predict([review_input])[0][0]

        label = 'High Quality' if pred_prob >= 0.5 else 'Low Quality'
        
        # Show result
        st.subheader(f'Prediction: {label}')
        st.text(f'Probability of being High Quality: {pred_prob:.2f}')
