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
    model.add(hub_layer)  # Directly add the hub layer
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

model = load_model()

# Load and preprocess your dataset
train_data = pd.read_csv('https://raw.githubusercontent.com/Furqan-Qureshi786/TextClassifier/main/wine-reviews.csv')

# Check if 'quality' is a binary column; for example, you may want to create it based on 'points'
# This step assumes quality is defined based on points, e.g., >= 85 is high quality
train_data['quality'] = np.where(train_data['points'] >= 85, 1, 0)  # Adjust this threshold as needed

# Prepare the data for training (split features and labels)
X = train_data['description']  # Feature column (text input)
y = train_data['quality']       # Target variable (binary)

# Streamlit app interface
st.title('Wine Review Quality Classifier')

# Text input for wine review
review_input = st.text_area('Enter your wine review:', '')

# Predict button
if st.button('Predict'):
    if review_input.strip() == '':
        st.error('Please enter a wine review.')
    else:
        # Preprocess the input (keep it as a string)
        review_input_array = np.array([review_input])

        # Prediction
        try:
            pred_prob = model.predict(review_input_array)  # Model expects numpy array
            pred_prob_value = pred_prob[0][0]  # Get the probability

            label = 'High Quality' if pred_prob_value >= 0.5 else 'Low Quality'
            
            # Show result
            st.subheader(f'Prediction: {label}')
            st.text(f'Probability of being High Quality: {pred_prob_value:.2f}')
        except Exception as e:
            st.error(f'Error during prediction: {e}')

