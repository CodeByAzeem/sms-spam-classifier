import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

def load_data():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/vectorizer.pkl", "rb") as f:
        tf = pickle.load(f)

    return model, tf

model, tf = load_data()

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("📩 SMS Spam Classifier")
st.caption("Detect spam messages using Machine Learning.")

st.divider()

input_sms = st.text_area(
    "Enter your message",
    placeholder="Type or paste your SMS here...",
    height=150
)

if st.button("🔍 Predict", use_container_width=True):

    if not input_sms.strip():
        st.warning("Please enter a message first.")

    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tf.transform([transformed_sms]).toarray()

        result = model.predict(vector_input)[0]

        st.divider()

        if result == 1:
            st.error("🚨 Spam Message Detected!")
            st.write("Be careful. This message may contain spam.")

        else:
            st.success("✅ Not Spam")
            st.write("This message appears to be legitimate.")

st.divider()

st.caption("Built with ❤️ using Python, NLTK and Streamlit")