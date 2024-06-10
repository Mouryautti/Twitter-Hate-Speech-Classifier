import streamlit as st

def load_model_data():
    # Simulate loading data
    model_name = "Hate Speech Detection Model"
    accuracy = 0.89
    precision = 0.78
    recall = 0.92
    return model_name, accuracy, precision, recall


model_name, accuracy, precision, recall = load_model_data()

st.title("About Us")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/X_logo.jpg/900px-X_logo.jpg", width=100)
st.sidebar.title("Model Information")
st.sidebar.write(f"Model Name: {model_name}")
st.sidebar.write(f"Accuracy: {accuracy}")
st.sidebar.write(f"Precision: {precision}")
st.sidebar.write(f"Recall: {recall}")
st.sidebar.title("About Hate Speech")
st.sidebar.write("Hate speech refers to speech that promotes or encourages hatred, violence, or discrimination against individuals or groups based on certain attributes such as race, religion, ethnicity, gender, sexual orientation, etc.")
st.sidebar.write("It can take various forms, including derogatory language, threats, slurs, or harassment.")
# Add a header and some text
st.header("Our Mission")
st.write("""
We aim to provide innovative solutions to detect and prevent hate speech online. Our team is dedicated to leveraging the power of machine learning and artificial intelligence to make the internet a safer place for everyone.
""")

# Add another header for the project guide
st.header("Our Project Guide")
# Add the project guide's photo (replace with an actual image URL or file path)
st.markdown('''
<div style="text-align: center;">
    <img src="https://media.licdn.com/dms/image/C5603AQH4I1B-cj0zsQ/profile-displayphoto-shrink_800_800/0/1616561894929?e=1723680000&v=beta&t=FMnfqv0eV0u85G4-C1uUE4gRCTf6tleAJIpcikP_Um8" width="300">
    <p><strong>Prof. Amaresh AM</strong></p>
    <p>Project Guide and Mentor</p>
</div>
''', unsafe_allow_html=True)
st.write("")
st.write("")

st.header("Our Team")
# Create a row of images for team members
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("https://media.licdn.com/dms/image/D5603AQE7DKqd2joj_w/profile-displayphoto-shrink_200_200/0/1687684622536?e=2147483647&v=beta&t=fVyomGdI3uqIxZVMlGvwB4zVPMQtAQkdH7nVDPWT5cg", caption="Tushar Singh", use_column_width=True)
with col2:
    st.image("https://via.placeholder.com/150x150.png?text=Charlie+Davis", caption="Charlie Davis", use_column_width=True)
with col3:
    st.image("https://media.licdn.com/dms/image/C4D03AQHwq-JPNHtf8Q/profile-displayphoto-shrink_200_200/0/1643511930170?e=2147483647&v=beta&t=_sg-qGfr-DLpUJJyJ7Q0vChdJ5jmhQHpl-RCnqaIiBc", caption="Rudraksh Kumar", use_column_width=True)
with col4:
    st.image("https://media.licdn.com/dms/image/D5603AQFHZnsRR8lRYg/profile-displayphoto-shrink_200_200/0/1672671260212?e=2147483647&v=beta&t=ZUaHMoSVBAeK-wn9JDROLCJHUX1o16DpyEjvLDTssy8", caption="VPS Konda Reddy", use_column_width=True)


# Add social media links
st.header("Links")
st.write("""
- [Github](https://github.com/theinit01/Twitter-Hate-Speech-Classifier)
""")

# Add footer
st.markdown('''
<hr style="border-top: 1px solid #bbb;">
<div style="text-align: center;">
    <p>&copy; 2024 Hate Speech Detection Team. All rights reserved.</p>
</div>
''', unsafe_allow_html=True)