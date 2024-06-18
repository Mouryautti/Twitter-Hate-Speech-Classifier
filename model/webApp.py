import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tweets import gather_tweets
import joblib
import useModel
import detoxModel
from detoxify import Detoxify
import time

#clf = joblib.load('hate_speech_classifier.pkl')
#cv = joblib.load('count_vectorizer.pkl')

def load_model_data():
    # Simulate loading data
    model_name = "Hate Speech Detection Model"
    accuracy = 0.89
    precision = 0.78
    recall = 0.92
    return model_name, accuracy, precision, recall


model_name, accuracy, precision, recall = load_model_data()

# Sidebar with model data and information about hate speech
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/X_logo.jpg/900px-X_logo.jpg", width=100)
st.sidebar.title("Model Information")
st.sidebar.write(f"Model Name: {model_name}")
st.sidebar.write(f"Accuracy: {accuracy}")
st.sidebar.write(f"Precision: {precision}")
st.sidebar.write(f"Recall: {recall}")
st.sidebar.title("About Hate Speech")
st.sidebar.write("Hate speech refers to speech that promotes or encourages hatred, violence, or discrimination against individuals or groups based on certain attributes such as race, religion, ethnicity, gender, sexual orientation, etc.")
st.sidebar.write("It can take various forms, including derogatory language, threats, slurs, or harassment.")
 
st.title('Hate Speech Classifier')
input_text = st.text_input('Enter a twitter handle:', '@')
 
username = input_text.lstrip('@')


if username != '' and '@' not in username:
    results = {}
    success_message = st.empty()

    print(f"Username: {username}")
    with st.spinner('Gathering tweets...'):
        tweets = gather_tweets(username)
    success_message.success('Tweets gathered successfully!')
    
    with st.spinner('Analyzing tweets...'):
        all_tweets_text = ""
        for tweet in tweets:
            all_tweets_text += tweet + " "
        tweets = [useModel.clean(tweet) for tweet in tweets]
        results = Detoxify('unbiased').predict(tweets)
        classified_tweets = detoxModel.classify_tweets(tweets, results)

        label_counts = {
                'Hate speech': 0,
                'toxicity': 0,
                'severe_toxicity': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_attack': 0,
                'No hate speech': 0
            }

        for classified_tweet in classified_tweets:
            label_counts[classified_tweet['classification']] += 1

            # Print the counts
        for label, count in label_counts.items():
            print(f"Number of tweets classified as '{label}': {count}")
            
            #st.write(label_counts)
            # Count the number of tweets in each category
        categories = list(results.keys())
        counts = [len(results[category]) for category in categories]

            # Create a bar graph
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(label_counts.keys(), label_counts.values())
        ax.set_xlabel('Category')
        ax.set_ylabel('Number of Tweets')
        ax.set_title('Number of Tweets per Category')
        ax.set_xticklabels(label_counts.keys(), rotation=45, ha='right')

        # Create a word cloud
        wordcloud = WordCloud(width=800, height=600, background_color='white').generate(all_tweets_text)
        wordcloud_fig, wordcloud_ax = plt.subplots(figsize=(8, 6))
        wordcloud_ax.imshow(wordcloud, interpolation='bilinear')
        wordcloud_ax.axis('off')
        wordcloud_ax.set_title('Word Cloud of Tweets')

        # Create a pie chart
        piefig, ax = plt.subplots(figsize=(8, 8))
        wedges, _, _ = ax.pie(label_counts.values(), autopct='%1.1f%%', startangle=140)

        # Set the legend
        ax.legend(label_counts.keys(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax.set_title('Distribution of Tweets by Category')

        legend = ax.legend(label_counts.keys(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(legend.get_texts(), fontsize='medium')

        st.pyplot(fig)
        st.pyplot(wordcloud_fig)
        st.pyplot(piefig)

        for label in label_counts.keys():
            tweets_for_label = [tweet['tweet'] for tweet in classified_tweets if tweet['classification'] == label]
            if tweets_for_label:
                st.header(f"Top Tweets for '{label}'")
                top_tweets = tweets_for_label[:3]  # Get up to 3 tweets
                for i, tweet in enumerate(top_tweets, start=1):
                    st.write(f"{i}. {tweet}")
            else:
                pass

    success_message.success('Tweets classified successfully!')
    time.sleep(5)  # Display for 5 seconds
    success_message.empty()  # Clear the success message slot