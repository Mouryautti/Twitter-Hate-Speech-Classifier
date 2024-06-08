import joblib
import re
import nltk
from nltk.corpus import stopwords 
import string

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load stopwords
stopwords = set(stopwords.words("english"))

# Load the model and CountVectorizer
clf = joblib.load('hate_speech_classifier.pkl')
cv = joblib.load('count_vectorizer.pkl')

# Clean function
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

# Function to predict a sentence
def predict_sentence(sentence, vectorizer, classifier):
    # Clean the sentence
    cleaned_sentence = clean(sentence)
    # Transform the cleaned sentence into a numerical representation
    vectorized_sentence = vectorizer.transform([cleaned_sentence])
    # Predict the label for the vectorized sentence
    predicted_label = classifier.predict(vectorized_sentence)
    return predicted_label[0]

data = ["you are good", "You are a piece of shit", "Muslims are the worst"]

for sentence in data:
    predicted_label = predict_sentence(sentence, cv, clf)
    print("Predicted Label:", predicted_label)
