# Spam Detection ML Project

This project is a Machine Learning model that classifies messages as Spam or Ham using Scikit-learn.

## Features
- Text preprocessing
- TF-IDF Vectorization
- Naive Bayes model
- Accuracy ~97%

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Numpy

## Dataset
Spam SMS dataset (CSV format)

## How to Run
1. Open the notebook (spam_model.ipynb)
2. Run all cells
3. Model will train and show accuracy

## Output
Model predicts whether a message is Spam or Not Spam

*COMPANY NAME*:CODTECH IT SOLUTIONS

*NAME*:SEJAL MAHATO

*INTERN ID*:CTIS3242

*DOMAIN NAME*:PYTHON

*MENTOR*:NEELA SANTOSH

# Spam Detection using Machine Learning

*This project focuses on building a predictive machine learning model to classify text messages as either spam or ham (not). The objective of this project is to demonstrate how Natural Language Processing (NLP) techniques can be applied to real-world problems like spam detection, which is widely used in email filtering systems and messaging platforms.
The dataset used in this project is a CSV file containing labeled messages. Each message is categorized as either "spam" or "ham". The project begins with data loading and preprocessing using Python libraries such as pandas and numpy. The preprocessing stage includes cleaning the text data by removing punctuation, converting text to lowercase, and eliminating unnecessary words. This step ensures that the data is in a suitable format for model training.
After preprocessing, the text data is transformed into numerical form using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique helps in converting textual information into meaningful numerical features that can be used by machine learning algorithms.
For the classification task, the Multinomial Naive Bayes algorithm is used. This algorithm is particularly effective for text classification problems due to its simplicity and efficiency. The dataset is split into training and testing sets to evaluate the performance of the model.
The model is trained on the training dataset and then tested on unseen data. The performance of the model is evaluated using accuracy metrics, and the model achieved an accuracy of approximately 97%, indicating a high level of effectiveness in distinguishing between spam and non-spam messages.
Additionally, the trained model and vectorizer are saved using pickle files, allowing the model to be reused without retraining. This makes the project more practical and closer to real-world applications.
This project demonstrates a complete machine learning workflow, including data preprocessing, feature extraction, model training, evaluation, and saving the model. It is a beginner-friendly yet industry-relevant implementation that highlights the importance of NLP in solving everyday problems.
Overall, this project provides a strong foundation for understanding text classification and can be further extended by integrating it into web applications or deploying it using APIs.*
