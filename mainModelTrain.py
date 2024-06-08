import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords 
import string
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load stopwords
stopwords = set(stopwords.words("english"))

df = pd.read_csv(r"C:\Users\tusha\OneDrive\Desktop\Twitter-Hate-Speech-Classifier\model\twitter_data.csv")

# Map class labels correctly
df['labels'] = df['class'].map({0: "Hate Speech Detected", 1: "Offensive language detected", 2: "No hate and Offensive speech"})

df = df[['tweet', 'labels']]
df.dropna(inplace=True)

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\. \S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join(word for word in text.split(' ') if word not in stopwords)
    return text
def predict_sentence(sentence, vectorizer, classifier):
    # Clean the sentence
    cleaned_sentence = clean(sentence)
    # Transform the cleaned sentence into a numerical representation
    vectorized_sentence = vectorizer.transform([cleaned_sentence])
    # Predict the label for the vectorized sentence
    predicted_label = classifier.predict(vectorized_sentence)
    return predicted_label[0]




df["tweet"] = df["tweet"].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Example sentence to predict
new_sentence = "I hate your country"

# Predict the label for the new sentence
predicted_label = predict_sentence(new_sentence, cv, clf)
print("Predicted Label:", predicted_label)

accuracy = accuracy_score(y_test, clf.predict(X_test))
report = classification_report(y_test, clf.predict(X_test))

print("Accuracy:", accuracy)
print("Report:", report)
