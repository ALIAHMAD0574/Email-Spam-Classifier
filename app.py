import streamlit as st
import  pickle
import  nltk
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def tranform_text(text):
  text = text.lower()
  # tokenization
  text = nltk.word_tokenize(text)
  # removing special charatcers  , stopwords and punctuation
  y = []
  for i in text:
    if (i.isalnum()) and (i not in stopwords.words('english')) and (i not in string.punctuation):
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))
  return ' '.join(y)


st.title('Email/Sms Spam Classifier')
sms = st.text_area('Enter the  message')

if st.button('Prdict'):
  # 1) preprocess

  tranform_sms = tranform_text(sms)
  # 2) vectorize
  vector_input = tfidf.transform([tranform_sms])
  # 3) predict
  result = model.predict(vector_input)[0]
  # 4) display
  if result == 1:
      st.header('Spam')
  else:
      st.header('Not Spam')


