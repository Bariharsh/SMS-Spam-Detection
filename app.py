import streamlit as st # type: ignore
import nltk
from nltk.corpus import stopwords
import pickle
import string
import pandas as pd
import requests
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
# Download resources once
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')  # NEW

def transform_text(text):
  #lower case
  text = text.lower()
  #Tokenization
  text = nltk.word_tokenize(text)
  y=[]
  #removing Special Charcetrs 
  for i in text:
    if i.isalnum():
      y.append(i)
  #Removing stop words
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model2.pkl','rb'))

st.title("Email/SMS Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    

    transformed_sms = transform_text(input_sms)
    vector_input = tfid.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")