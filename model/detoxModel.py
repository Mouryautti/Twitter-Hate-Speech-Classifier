from detoxify import Detoxify
from tweets import gather_tweets
from useModel import clean

def classify_tweets(tweets, results, threshold_hate=0.5, threshold_no_hate=0.003):
    labels = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    classified_tweets = []

    for i, tweet in enumerate(tweets):
        # Extract probabilities for the current tweet
        tweet_probabilities = {label: results[label][i] for label in labels}
        # Determine the label with the highest probability
        max_label = max(tweet_probabilities, key=tweet_probabilities.get)
        max_probability = tweet_probabilities[max_label]

        # Label as "Hate speech" if the highest probability is above the hate threshold
        if max_probability > threshold_hate:
            classification = 'Hate speech'
        # Label as "No hate speech" if the highest probability is below the no hate threshold
        elif max_probability < threshold_no_hate:
            classification = 'No hate speech'
        else:
            classification = max_label

        classified_tweets.append({
            'tweet': tweet,
            'classification': classification,
            'probability': max_probability
        })
    
    return classified_tweets

if __name__ == '__main__':

    # Gather tweets
    username = input("Enter username: ")
    tweets = gather_tweets(username)

    # Clean tweets
    tweets = [clean(tweet) for tweet in tweets]

    # Get predictions
    results = Detoxify('unbiased').predict(tweets)

    # Classify tweets
    classified_tweets = classify_tweets(tweets, results)

    # Print classified tweets
    for classified_tweet in classified_tweets:
        print(f"Tweet: {classified_tweet['tweet']}\nClassification: {classified_tweet['classification']}\nProbability: {classified_tweet['probability']}\n")
