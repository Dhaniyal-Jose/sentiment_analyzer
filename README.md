# Sentiment Analysis Project

This project is a sentiment analysis tool that uses a deep learning model to classify text as positive, negative, or neutral. The model is trained on a dataset of text samples and can be used to predict the sentiment of new text.

## Project Structure

```
sentiment_analyzer/
├── data/
│   └── sentiment_analysis.csv
├── models/
│   ├── label_encoder.pickle
│   ├── sentiment_model.h5
│   ├── sentiment_model_v3.h5
│   ├── sentiment_model_weighted.h5
│   └── tokenizer.pickle
├── notebooks/
│   └── sentiment_analysis.ipynb
├── scripts/
│   ├── check_libs.py
│   ├── download_stopwords.py
│   ├── inspect_data.py
│   ├── predict.py
│   ├── train_sentiment_model.py
│   ├── train_sentiment_model_v3.py
│   └── train_sentiment_model_weighted.py
└── utils/
    └── preprocess.py
```

- **data/**: Contains the dataset used for training and evaluation.
- **models/**: Stores the trained models, tokenizer, and label encoder.
- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
- **scripts/**: Contains Python scripts for various tasks such as checking dependencies, downloading data, training models, and making predictions.
- **utils/**: Contains utility functions for preprocessing text data.

## Dataset

The dataset used in this project is `sentiment_analysis.csv`, which contains text content and corresponding sentiment labels.

## Dependencies

The project requires the following Python libraries:

- pandas
- scikit-learn
- tensorflow
- nltk

You can check if these libraries are installed by running the `check_libs.py` script:

```bash
python scripts/check_libs.py
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd sentiment_analyzer
    ```

2.  **Install the dependencies:**

    ```bash
    pip install pandas scikit-learn tensorflow nltk
    ```

3.  **Download the NLTK stopwords:**

    ```bash
    python scripts/download_stopwords.py
    ```

## Usage

### Inspecting the Data

To inspect the class distribution of the dataset, run the following command:

```bash
python scripts/inspect_data.py
```

### Training the Model

There are three different training scripts available:

1.  **`train_sentiment_model.py`**: Trains a simple LSTM model.

    ```bash
    python scripts/train_sentiment_model.py
    ```

2.  **`train_sentiment_model_weighted.py`**: Trains an LSTM model with class weights to handle class imbalance. This also saves the tokenizer and label encoder.

    ```bash
    python scripts/train_sentiment_model_weighted.py
    ```

3.  **`train_sentiment_model_v3.py`**: Trains a more advanced LSTM model with text cleaning, class weights, and a learning rate scheduler.

    ```bash
    python scripts/train_sentiment_model_v3.py
    ```

### Making Predictions

To make a sentiment prediction on a sample sentence, run the `predict.py` script:

```bash
python scripts/predict.py
```

This will load the `sentiment_model_weighted.h5` model and predict the sentiment of a hardcoded sample sentence.

## Models

This project includes three different trained models:

-   **`sentiment_model.h5`**: A simple LSTM model.
-   **`sentiment_model_weighted.h5`**: An LSTM model trained with class weights to address class imbalance.
-   **`sentiment_model_v3.h5`**: A more advanced LSTM model with better preprocessing and training techniques.

The `predict.py` script uses the `sentiment_model_weighted.h5` model by default.

## Future Improvements

-   Implement a web interface or API to make predictions on user-provided text.
-   Experiment with different model architectures (e.g., GRU, Bi-LSTM, Transformers).
-   Use pre-trained word embeddings (e.g., GloVe, Word2Vec).
-   Perform hyperparameter tuning to optimize the model's performance.
-   Add support for more languages.
