# IMDb Movie Review Sentiment Analysis

A deep learning-based Natural Language Processing project that classifies IMDb movie reviews as positive or negative using a Bidirectional LSTM neural network.

## Project Overview

This project uses the IMDb movie review dataset containing 50,000 labeled movie reviews. A Bidirectional LSTM model is trained to learn sentiment patterns from the reviews and classify unseen reviews as either positive or negative.

The project also includes a Streamlit web application that allows users to enter a movie review and receive a sentiment prediction.

## Features

- Text preprocessing and cleaning
- HTML tag removal
- Lowercase conversion
- Removal of unwanted characters
- Tokenization using Keras Tokenizer
- Sequence padding
- Word embeddings
- Bidirectional LSTM for sentiment classification
- Binary sentiment prediction
- Streamlit web application for interactive predictions

## Dataset

The project uses the IMDb Dataset containing:

- 50,000 movie reviews
- 25,000 positive reviews
- 25,000 negative reviews

Each review is labeled as either `positive` or `negative`.

## Data Preprocessing

The reviews are cleaned before being passed to the model.

The preprocessing steps include:

1. Removing HTML tags
2. Converting text to lowercase
3. Removing unwanted characters
4. Replacing removed characters with spaces
5. Removing extra whitespace
6. Tokenizing the reviews
7. Converting reviews into numerical sequences
8. Padding sequences to a maximum length of 600

The tokenizer uses the 10,000 most frequent words from the training data.

## Model Architecture

The sentiment classifier uses the following architecture:

```text
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

The model uses a sigmoid output layer because this is a binary classification problem.

## Model Training

The dataset is divided into training and testing sets.

The tokenizer is fitted only on the training data to avoid data leakage. The same tokenizer is then used to transform both training and testing reviews.

Early stopping is used during training to reduce overfitting and restore the model weights from the best validation performance.

## Project Structure

```text
IMDb-Sentiment-Analysis/
|
├── app.py
├── sentiment_analysis.ipynb
├── sentiment_model.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── text_preprocessing_utils.py
├── requirements.txt
└── README.md
```

## Streamlit Application

The project includes a Streamlit application where users can enter their own movie review.

The application performs the following steps:

```text
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
Sentiment Prediction
```

The application returns either `Positive` or `Negative` along with the prediction confidence.

## Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- BeautifulSoup
- Regular Expressions
- Streamlit
- Matplotlib
- Seaborn

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/IMDb-Sentiment-Analysis.git
```

Navigate to the project directory:

```bash
cd IMDb-Sentiment-Analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## Example

Input:

```text
I really enjoyed this movie. The acting was excellent and the story was very engaging.
```

Output:

```text
Positive
```

Input:

```text
The movie was extremely boring and the story was poorly written. I would not recommend it.
```

Output:

```text
Negative
```

## Limitations

The model may have difficulty understanding sarcasm, irony, and reviews where the intended sentiment differs from the literal meaning of the words.

## Future Improvements

- Compare Bidirectional LSTM with a standard LSTM
- Experiment with different vocabulary sizes
- Experiment with different sequence lengths
- Try pretrained word embeddings
- Experiment with GRU and Transformer-based models
- Improve handling of sarcasm and contextual sentiment
- Deploy the Streamlit application

## Author

Riddhima Mehra
