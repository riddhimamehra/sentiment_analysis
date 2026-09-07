# CineSense: Movie Review Sentiment Classifier

A deep learning-based NLP project that classifies IMDb movie reviews as positive or negative using a Bidirectional LSTM neural network, benchmarked against a classical ML baseline.

🔗 **Live demo:** https://sentimentanalysis-afpuezedy5mq4pkvmofenx.streamlit.app/

## Project Overview

This project uses the IMDb movie review dataset (50,000 labeled reviews) to train a Bidirectional LSTM model that classifies unseen reviews as positive or negative. To validate that the deep learning approach is actually worth its added complexity, its performance is benchmarked against a TF-IDF + Logistic Regression baseline.

The project includes a Streamlit web application where users can enter a review and get a live sentiment prediction with confidence score.

## Features

- Text preprocessing and cleaning (HTML tag removal, lowercasing, noise removal)
- Tokenization using Keras Tokenizer, with sequence padding
- Word embeddings + Bidirectional LSTM for sentiment classification
- **Baseline comparison against TF-IDF + Logistic Regression** to justify the deep learning approach with evidence, not just architecture choice
- Streamlit web application for interactive predictions with confidence score

## Dataset

- 50,000 IMDb movie reviews
- 25,000 positive / 25,000 negative
- Labeled `positive` / `negative`

## Data Preprocessing

1. Remove HTML tags
2. Convert to lowercase
3. Remove unwanted characters (replaced with spaces)
4. Remove extra whitespace
5. Tokenize reviews
6. Convert to numerical sequences
7. Pad sequences to a maximum length of 600

The tokenizer uses the 10,000 most frequent words from the training data, fitted only on the training set to avoid data leakage.

## Model Architecture (CineSense)

```
Input Review
     |
Text Preprocessing
     |
Tokenization
     |
Sequence Padding
     |
Embedding Layer
     |
Bidirectional LSTM
     |
Dropout
     |
Dense Layer
     |
Sigmoid Output
     |
Positive / Negative
```

Early stopping is used during training to reduce overfitting and restore the best validation weights.

## Model Comparison: BiLSTM vs. TF-IDF + Logistic Regression

To confirm the deep learning model earns its complexity, its performance is compared against a simpler classical baseline on the same train/test split.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 89.92% | 89.51% | 90.44% | 89.97% |
| Bidirectional LSTM (CineSense) | 88.88% | 89.45% | 88.16% | 88.80% |

**Takeaway:** Contrary to the usual assumption that deep learning outperforms classical ML, the TF-IDF + Logistic Regression baseline slightly outperformed the BiLSTM on every metric here (~1 point higher accuracy and F1). This is a known pattern on IMDb-style sentiment data: reviews are long enough and lexically distinct enough (words like "boring" or "brilliant" are strong standalone signals) that a linear model over sparse word features captures most of the useful signal, while the BiLSTM's sequential modeling of word order adds complexity without a proportional accuracy gain on this dataset. The BiLSTM would likely show its advantage more clearly on sentences with negation, sarcasm, or long-range dependencies — an area for further error analysis (see Future Improvements).

See `sentiment_analysis.ipynb` for the full comparison code, including a confusion matrix for both models.

## Project Structure

```
CineSense/
|
├── app.py
├── sentiment_analysis.ipynb
├── sentiment_model.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── text_preprocessing_utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Streamlit Application

```
User Review
     |
Clean Text
     |
Tokenizer
     |
Sequence Padding
     |
Bidirectional LSTM
     |
Sentiment Prediction + Confidence
```

## Technologies Used

Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, BeautifulSoup, Regular Expressions, Streamlit, Matplotlib, Seaborn

## Installation

```
git clone https://github.com/riddhimamehra/sentiment_analysis.git
cd sentiment_analysis
pip install -r requirements.txt
```

## Running the Application

```
streamlit run app.py
```

## Example

**Input:** "I really enjoyed this movie. The acting was excellent and the story was very engaging."
**Output:** Positive

**Input:** "The movie was extremely boring and the story was poorly written. I would not recommend it."
**Output:** Negative

## Limitations

- May struggle with sarcasm, irony, and reviews where intended sentiment differs from literal wording
- Trained and evaluated only on IMDb-style reviews — may not generalize well to other domains (e.g., product reviews, social media text)

## Future Improvements

- Error analysis: identify specific reviews where the BiLSTM correctly classifies sentiment but the baseline fails (and vice versa) — e.g., cases involving negation ("not bad at all") or sarcasm — to isolate where sequential modeling actually helps
- Experiment with different vocabulary sizes and sequence lengths
- Try pretrained word embeddings (GloVe, Word2Vec) to see if the BiLSTM can close the gap with better initialization
- Experiment with GRU and Transformer-based models
- Improve handling of sarcasm and contextual sentiment

## Author

Riddhima Mehra — [GitHub](https://github.com/riddhimamehra)
