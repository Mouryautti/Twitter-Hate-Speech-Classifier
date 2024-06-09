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
import joblib

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load stopwords
stopwords = set(stopwords.words("english"))

df = pd.read_csv(r"C:\Users\tusha\OneDrive\Desktop\Twitter-Hate-Speech-Classifier\model\twitter_data.csv")

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


df["tweet"] = df["tweet"].apply(clean)

# Features and labels
x = np.array(df["tweet"])
y = np.array(df["labels"])

# Vectorize text
cv = CountVectorizer()
x = cv.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the model and CountVectorizer
# joblib.dump(clf, 'hate_speech_classifier.pkl')
# joblib.dump(cv, 'count_vectorizer.pkl')

# Evaluate model
accuracy = accuracy_score(y_test, clf.predict(X_test))
report = classification_report(y_test, clf.predict(X_test))

print("Accuracy:", accuracy)
print("Report:", report)