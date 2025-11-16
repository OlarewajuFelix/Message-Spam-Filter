Spam Filter Machine Learning Project

This project implements two different machine learning models for
detecting spam messages using the Spambase dataset:

-   Logistic Regression (spam_filter.py)
-   Naive Bayes (GaussianNB) (spam_filter_naive_bayes.py)

Both scripts train, evaluate, and save a predictive model.

Project Structure

spam_filter.py spam_filter_naive_bayes.py spambase_csv.csv README.txt

Features

-   Loads and analyzes the Spambase dataset
-   Splits data into training/testing subsets
-   Trains two different machine learning models
-   Provides accuracy, classification report, and confusion matrix
-   Saves each trained model using joblib

Requirements

pip install pandas scikit-learn joblib

Running the Models

1.  Logistic Regression: python spam_filter.py Output:
    spam_filter_model.pkl

2.  Naive Bayes: python spam_filter_naive_bayes.py Output:
    spam_filter_naive_bayes.pkl

Dataset

Uses the Spambase dataset in CSV format. - ‘class’ column is the target
(1 = spam, 0 = not spam) - All other columns are features

Notes

-   Naive Bayes is fast but may be less accurate.
-   Logistic Regression tends to perform better with high-dimensional
    numeric data.
