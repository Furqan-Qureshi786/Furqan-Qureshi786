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
    
    return model

model = load_model()

# Streamlit app interface
st.title('Wine Review Quality Classifier')

# Upload training data
uploaded_file = st.file_uploader("Upload training data (CSV)", type=["csv"])

if uploaded_file is not None:
    train_data = pd.read_csv(uploaded_file)
    
    # Ensure your specific column is numeric
    train_data['your_column_name'] = pd.to_numeric(train_data['your_column_name'], errors='coerce')
    train_data.fillna(0, inplace=True)  # or use train_data.dropna(inplace=True)

    # Separate features and labels
    X_train = train_data['review_column_name']  # replace with your actual review column name
    y_train = train_data['label_column_name']    # replace with your actual label column name

    # Fit the model
    history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Text input for wine review
review_input = st.text_area('Enter your wine review:', '')

# Predict button
if st.button('Predict'):
    if review_input.strip() == '':
        st.error('Please enter a wine review.')
    else:
        # Ensure the input is a string
        review_input = str(review_input)
        
        # Prediction
        pred_prob = model.predict([review_input])[0][0]

        label = 'High Quality' if pred_prob >= 0.5 else 'Low Quality'
        
        # Show result
        st.subheader(f'Prediction: {label}')
        st.text(f'Probability of being High Quality: {pred_prob:.2f}')
