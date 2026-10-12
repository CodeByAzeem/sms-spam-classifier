import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

st.title('📩 SMS Spam Classifier')

st.write("Enter an SMS message below to check whether it is spam.")

input_sms = st.text_area('Enter the Message: ')

if st.button('Predict'):

    if not input_sms.strip():
        st.warning("Please enter a message first.")

    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tf.transform([transformed_sms])
        result = model.predict(vector_input.toarray())[0]

        if result == 1:
            st.header('🚨 Spam')
        else:
            st.header('✅ Not Spam')