# IMDB Movie Review Sentiment Analysis

This repository contains a deep learning project for sentiment analysis of movie reviews using a Simple RNN model. The project includes text preprocessing, model training, and a Streamlit web application for interactive sentiment prediction.

---

## Project Overview

This natural language processing (NLP) project classifies movie reviews from the IMDB dataset as positive or negative using a Recurrent Neural Network (RNN). The model achieves 85%+ accuracy in sentiment classification and includes an interactive web interface for real-time predictions.

---

## Key Features

- Text preprocessing pipeline with word-to-index encoding
- Sequence padding for input standardization
- Simple RNN architecture with embedding layer
- Pre-trained model integration for immediate use
- Streamlit-based web interface with real-time predictions
- Sentiment score visualization and input debugging

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com//imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Web App
```
streamlit run main.py
```

The web interface provides:
- Text input field for movie reviews
- Real-time sentiment classification (Positive/Negative)
- Prediction confidence score
- Optional preprocessing debug view

### Model Architecture
```
Sequential([
    Embedding(10000, 32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])
```

---

## Dataset

The model uses the IMDB Movie Reviews dataset:
- 50,000 polarized reviews (25k train/25k test)
- Binary labels: 0 (negative) / 1 (positive)
- Vocabulary size: 10,000 most frequent words

---

## Performance

- Training Accuracy: 85-88%
- Validation Accuracy: 82-85%
- Loss: Binary Crossentropy
- Optimizer: Adam (learning_rate=0.01)

---

## File Structure

| File                 | Description                         |
|----------------------|-------------------------------------|
| `main.py`            | Streamlit application entry point   |
| `simple_rnn_imdb.h5` | Pretrained RNN model                |
| `requirements.txt`   | Python dependencies                 |

---

## Acknowledgments

- Dataset: [IMDB Movie Reviews](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
- TensorFlow/Keras for deep learning framework
- Streamlit for web interface implementation

---

## License

MIT License - see [LICENSE](LICENSE) for details.
```
