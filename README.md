# Hate Speech Classifier
<div align = "center">
    <img src="https://github.com/theinit01/Twitter-Hate-Speech-Classifier/blob/main/assets/ScreenRecording2024-08-05154029-ezgif.com-video-to-gif-converter.gif" alt="Animated GIF" />
</div>

This repository contains a Streamlit web app for classifying tweets using a hate speech detection model. Users can input a Twitter handle, and the app gathers tweets associated with that handle. It then classifies the tweets into different categories using a machine learning model trained to detect hate speech.

## Features

- Allows users to input a Twitter handle.
- Gathers tweets associated with the input handle.
- Classifies tweets into categories using a hate speech detection model.
- Displays the number of tweets per category in a bar graph.
- Generates a word cloud visualization of the collected tweets.
- Shows a pie chart displaying the distribution of tweets by category.

## Setup

To run the Streamlit app locally, follow these steps:

1. Clone this repository:

    ```
    git clone https://github.com/theinit01/Twitter-Hate-Speech-Classifier
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```
    streamlit run model/webApp.py
    ```

## Usage

1. Enter a Twitter handle in the input field provided.
2. Wait for the app to gather tweets associated with the entered handle.
3. View the classification results in the form of a bar graph, word cloud, and pie chart.
4. Explore the sidebar for information about the hate speech detection model and hate speech.

## About

This project was developed as a part of our Mini Project. It demonstrates the use of machine learning for hate speech detection and provides an interactive web interface for users to explore the classification results.


## License

This project is licensed under the [MIT License](LICENSE).
