import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tweets import gather_tweets
import joblib
import useModel
 
clf = joblib.load('hate_speech_classifier.pkl')
cv = joblib.load('count_vectorizer.pkl')

def load_model_data():
    # Simulate loading data
    model_name = "Hate Speech Detection Model"
    accuracy = 0.85
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

    print(f"Username: {username}")
    with st.spinner('Gathering tweets...'):
        tweets = gather_tweets(username)
    st.success('Tweets gathered successfully!')
    all_tweets_text = ""
    for tweet in tweets:
        predicted_label = useModel.predict_sentence(tweet, cv, clf)
        if predicted_label not in results:
            results[predicted_label] = []
        results[predicted_label].append(tweet)
        all_tweets_text += " " + tweet

    '''
    # Count the number of tweets in each category
    categories = list(results.keys())
    counts = [len(results[category]) for category in categories]

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust the figure size for smaller graph
    ax.bar(categories, counts)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Tweets')
    ax.set_title('Number of Tweets per Category')
    ax.set_xticklabels(categories, rotation=45, ha='right')  # Rotate labels for better readability

    # Create a word cloud
    wordcloud = WordCloud(width=480, height=320, background_color='white').generate(all_tweets_text)
    wordcloud_fig, wordcloud_ax = plt.subplots(figsize=(5, 4))  # Adjust the figure size for smaller word cloud
    wordcloud_ax.imshow(wordcloud, interpolation='bilinear')
    wordcloud_ax.axis('off')
    wordcloud_ax.set_title('Word Cloud of Tweets')

    # Create a pie chart
    piefig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figure size
    ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribution of Tweets by Category')

    # Display the visualizations side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(fig)
    with col2:
        st.pyplot(piefig)
    with col3:
        st.pyplot(wordcloud_fig)

        '''
    
    st.write(results)