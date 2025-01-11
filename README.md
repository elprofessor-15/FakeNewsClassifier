# FakeNewsClassifier
This repository contains the implementation of a Fake News Classifier using machine learning techniques. The objective of the project is to classify news articles into real or fake categories based on the content of the news title. The project utilizes the Kaggle Fake News Dataset for training and testing the models.
# Fake News Classifier

This project implements a machine learning-based fake news detection system that classifies news articles as **Real** or **Fake** based on their titles.

## Dataset

The dataset used for this project is the **Kaggle Fake News Dataset**, which contains news titles labeled as real or fake.

- **Training Data**: `kaggle_fake_train.csv`
- **Test Data**: `kaggle_fake_test.csv`

## Libraries Used

- `numpy` for numerical operations
- `pandas` for data manipulation
- `nltk` for text preprocessing
- `scikit-learn` for machine learning models and evaluation metrics
- `matplotlib` and `seaborn` for data visualization

## Steps:

1. **Data Preprocessing**:
   - Dropped the unnecessary 'id' column.
   - Handled missing values and reset indices.
   - Tokenized the news titles, removed stopwords, and performed stemming.

2. **Feature Extraction**:
   - Used Bag-of-Words with 1-3 n-grams to represent text data.

3. **Model Training**:
   - Trained two models: **Multinomial Naive Bayes** and **Logistic Regression**.
   - Evaluated the models using accuracy, precision, recall, and confusion matrix.

4. **Hyperparameter Tuning**:
   - Tuned the models using different hyperparameters to optimize performance.

5. **Prediction**:
   - Implemented a prediction function to classify new news articles as fake or real.

## How to Use

1. Clone the repository:
   git clone https://github.com/elprofessor-15/FakeNewsClassifier.git
2. Install the required dependencies:
   pip install -r requirements.txt
3. Run the `fake_news_classifier.ipynb` file to train the model and make predictions.
   
   
