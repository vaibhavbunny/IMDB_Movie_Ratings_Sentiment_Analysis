# 🎬 IMDB_Movie_Ratings_Sentiment_Analysis

A sentiment analysis application that classifies movie reviews from the IMDB dataset as **positive** or **negative** using a **Simple RNN model** in TensorFlow/Keras, with an interactive web interface built using **Streamlit**.

---

## 🧠 Project Highlights

- Uses the **IMDB dataset** built into Keras.
- Applies **Simple RNN architecture** for sequence modeling.
- Allows **real-time sentiment prediction** via web app.
- Interactive frontend using **Streamlit**.
- Pretrained model saved as `simple_rnn_imdb.h5`.

---

## 📁 Project Structure

```

.
├── main.py                                # Streamlit app for review classification
├── embedding.ipynb                        # Embedding exploration and visualization
├── prediction.ipynb                       # Prediction utilities and testing
├── simplernn.ipynb                        # Model training notebook
├── simple\_rnn\_imdb.h5                     # Trained RNN model
├── simple\_rnn\_imdb\_weights.weights.h5     # Weights of the trained model
├── README.md                              # Project documentation

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/IMDB_Movie_Ratings_Sentiment_Analysis.git
cd IMDB_Movie_Ratings_Sentiment_Analysis
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

> Or manually install:

```bash
pip install tensorflow streamlit numpy
```

### 3. Run the Streamlit App

```bash
streamlit run main.py
```

---

## ✨ Features

* ✅ Real-time input of movie reviews
* ✅ Text preprocessing (lowercasing, tokenization, padding)
* ✅ Prediction using pretrained RNN model
* ✅ User-friendly Streamlit interface

---

## ⚙️ How It Works

1. Input review is taken from user.
2. Words are tokenized using the original IMDB word index.
3. Encoded sequence is padded to match input length (maxlen = 500).
4. Loaded RNN model (`simple_rnn_imdb.h5`) predicts the sentiment score.
5. If score > 0.5 → **Positive**, else → **Negative**

---

## 📚 Dataset

* **IMDB Movie Reviews Dataset** from Keras:

  * 50,000 reviews, labeled positive or negative
  * Pre-tokenized and indexed
  * Split into 25k training and 25k test samples

---

## 📦 Model

* Model architecture:

  * `Embedding` layer
  * `SimpleRNN` layer
  * `Dense` output with sigmoid activation
* Trained using:

  * Binary crossentropy loss
  * Adam optimizer
  * Accuracy metric

---

## 📊 Notebooks

* `embedding.ipynb`: Understand embeddings
* `simplernn.ipynb`: Training code for the RNN model
* `prediction.ipynb`: Exploratory prediction utilities

---

## 🖥️ Deployment

The Streamlit app is lightweight and can be deployed to:

* **Streamlit Community Cloud**
* **Heroku**
* **Local servers or containers**

---
