import joblib
import re
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already downloaded
#nltk.download('stopwords')

# Load stopwords
#stopwords = set(stopwords.words("english"))

# Load the model and CountVectorizer

# Clean function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
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



if __name__ == "__main__":
    '''
    Testing Purpose
    '''
    data = ["you are good", "You are a piece of shit", "Muslims are the worst"]

    # Dictionary to store classified tweets
    classified_tweets = {}

    # Classify each tweet and store in the dictionary
    for sentence in data:
        predicted_label = predict_sentence(sentence, cv, clf)
        if predicted_label not in classified_tweets:
            classified_tweets[predicted_label] = []
        classified_tweets[predicted_label].append(sentence)

    # Print the dictionary of classified tweets
    print(classified_tweets)