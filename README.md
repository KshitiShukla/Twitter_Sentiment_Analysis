# Twitter_Sentiment_Analysis

## Project Overview
This project is a Natural Language Processing (NLP) challenge focused on Sentiment Analysis to detect hate speech in social media posts. The core objective is to build a machine learning model that can effectively classify tweets as either containing racist/sexist sentiment (Label '1') or being safe/non-offensive (Label '0').

This solution involves a complete data science pipeline, from cleaning noisy text data to developing and evaluating classification models.

## Key Features & Goals
- Objective: Classify tweets into two categories: Hate Speech (1) and Non-Hate Speech (0).
- Data Preparation: Robust pre-processing of raw tweet text to handle noise like special characters, punctuation, and non-essential terms.
- Feature Engineering: Conversion of cleaned text into numerical features using advanced vectorization techniques.
- Model Evaluation: Performance is primarily measured using the F1-Score, a key metric for classification problems with imbalanced datasets.

## Technology Stack & Libraries
The project is implemented in Python and leverages the following core libraries:
## Data & NLP
- pandas and numpy: For data manipulation and numerical operations.

- re (Regex): Essential for efficient text cleaning and pattern matching.

- nltk (Natural Language Toolkit): Used for tokenization, stemming, and managing stopwords.

- Vectorizers: CountVectorizer (Bag-of-Words) and TfidfVectorizer (TF-IDF) for converting text into numerical feature vectors.

## Visualization
- matplotlib and seaborn: For data exploration and visualization (e.g., word clouds, common word analysis).

- wordcloud: To visualize the most frequent words in the dataset for both positive and negative classes.

## Machine Learning Models
- sklearn (Scikit-learn): Used for standard machine learning tasks, including model selection and evaluation.

- LogisticRegression: A strong baseline classification model.

- SVC (Support Vector Classifier): A robust kernel-based classification algorithm.

- XGBClassifier (XGBoost): A high-performance gradient-boosting framework for advanced classification.

## Project Steps & Methodology
The solution follows a structured approach to tackle the NLP classification problem:

**1. Data Loading & Initial Exploration**
Loading the dataset of tweets and their corresponding labels.

Performing an initial analysis of the data structure and label distribution.

**2. Tweets Preprocessing & Cleaning**
Noise Removal: Removing Twitter-specific noise like mentions (@user), URLs, hashtags symbols (#), and special characters.

Tokenization & Normalization: Converting text to lowercase, tokenizing the text, and applying techniques like stemming (or lemmatization) to reduce words to their root form.

**3. Exploratory Data Analysis (EDA)**
Visualizing the most common words and trends in the entire dataset.

Separately analyzing word frequencies and hashtags associated with the two sentiment classes (Hate Speech vs. Non-Hate Speech) to gain contextual insights.

**4. Feature Extraction**
Applying Bag-of-Words and TF-IDF to transform the cleaned text into numerical feature matrices.

**5. Model Training & Evaluation**
Splitting the data into training and validation sets.

Training various classification models (LogisticRegression, SVC, XGBClassifier) using the extracted features.

Evaluating model performance, with a primary focus on maximizing the F1-Score to ensure balanced precision and recall for the target class (Hate Speech).
