import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

# Title for the Streamlit App
st.title("Wine Review Quality Classifier")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("wine-reviews.csv", usecols=['country', 'description', 'points', 'price', 'variety', 'winery'])
    df = df.dropna(subset=["description", "points"])
    return df

df = load_data()
st.write("Data Preview:")
st.write(df.head())

# Show histogram for points
st.subheader("Points Histogram")
fig, ax = plt.subplots()
ax.hist(df.points, bins=20)
ax.set_title("Points histogram")
ax.set_ylabel("N")
ax.set_xlabel("Points")
st.pyplot(fig)

# Preprocessing
df["label"] = (df.points >= 90).astype(int)
df = df[["description", "label"]]

train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
    df = dataframe.copy()
    labels = df.pop('label')
    df = df["description"]
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_data = df_to_dataset(train)
valid_data = df_to_dataset(val)
test_data = df_to_dataset(test)

# Embedding Model
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

# Build Model
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

# Train the Model
history = model.fit(train_data, epochs=5, validation_data=valid_data)

# Evaluate the model
st.subheader("Model Evaluation")
train_loss, train_acc = model.evaluate(train_data)
val_loss, val_acc = model.evaluate(valid_data)
test_loss, test_acc = model.evaluate(test_data)

st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Validation Accuracy: {val_acc:.2f}")
st.write(f"Test Accuracy: {test_acc:.2f}")

# Prediction on user input
st.subheader("Try It Yourself!")
user_input = st.text_area("Enter a wine description to classify:")
if st.button("Predict"):
    if user_input:
        pred = model.predict([user_input])
        prediction = "High Quality" if pred[0] >= 0.5 else "Low Quality"
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter a description.")
