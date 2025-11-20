# Twitter_Sentiment_Analysis

## Project Overview

This project is a Natural Language Processing (NLP) challenge focused on Sentiment Analysis to detect hate speech in social media posts. The core objective is to build a machine learning model that can effectively classify tweets as either containing racist/sexist sentiment (Label '1') or being safe/non-offensive (Label '0').

This solution involves a complete data science pipeline, from cleaning noisy text data to developing and evaluating classification models.

## Key Features & Goals

Objective: Classify tweets into two categories: Hate Speech (1) and Non-Hate Speech (0).

Data Preparation: Robust pre-processing of raw tweet text to handle noise like special characters, punctuation, and non-essential terms.

Feature Engineering: Conversion of cleaned text into numerical features using advanced vectorization techniques.

Model Evaluation: Performance is primarily measured using the F1-Score, a key metric for classification problems with imbalanced datasets.

## Project Workflow & Methodology

The solution follows a structured approach to tackle the NLP classification problem:

1. Data Loading & Initial Exploration

Loading the dataset of tweets and their corresponding labels.

Performing an initial analysis of the data structure and label distribution.

2. Tweets Preprocessing & Cleaning

Noise Removal: Removing Twitter-specific noise like mentions (@user), URLs, hashtags symbols (#), and special characters.

Tokenization & Normalization: Converting text to lowercase, tokenizing the text, and applying techniques like stemming (or lemmatization) to reduce words to their root form.

3. Exploratory Data Analysis (EDA)

Visualizing the most common words and trends in the entire dataset.

Separately analyzing word frequencies and hashtags associated with the two sentiment classes (Hate Speech vs. Non-Hate Speech) to gain contextual insights.

4. Feature Extraction

Transforming the cleaned text into numerical features that machine learning models can process. Two primary methods are used:

Bag-of-Words (CountVectorizer)

TF-IDF (TfidfVectorizer)

5. Model Training & Evaluation

Training various classification models: LogisticRegression, SVC (Support Vector Classifier), and XGBClassifier (XGBoost), using the extracted features.

Evaluating model performance, with a primary focus on maximizing the F1-Score.

## Technical Dependencies & Environment

Category

Tools / Libraries Used

Data & NLP

pandas, numpy, re (Regex), nltk, CountVectorizer, TfidfVectorizer

Machine Learning

sklearn (Scikit-learn), LogisticRegression, SVC, XGBClassifier

Visualization

matplotlib, seaborn, wordcloud

## Conclusion & Future Work

This project demonstrates a robust, end-to-end framework for classifying social media text based on sentiment. The current performance provides a strong baseline for hate speech detection.

How You Can Contribute:

We welcome contributions to further enhance the model's accuracy and capabilities! Areas for improvement include:

Exploring Deep Learning: Integrating techniques like Word2Vec or BERT embeddings.

Hyperparameter Optimization: Fine-tuning the parameters of the SVC or XGBoost models.

Advanced Preprocessing: Implementing more sophisticated techniques like Lemmatization instead of stemming.

Feel free to fork this repository, submit pull requests, or open an issue to discuss new ideas!
