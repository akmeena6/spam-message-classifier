import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the TF-IDF vectorizer and preferred model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
preferred_model = pickle.load(open('preferred_model.pkl', 'rb'))

# Streamlit app title and input box
st.title("SMS/Email Spam Classifier")
Input_sms = st.text_area("Enter the message")

# Function to preprocess the input SMS
def usable_sms(sms):
    # Convert to lowercase and tokenize
    sms = sms.lower()
    sms = nltk.word_tokenize(sms)

    # Remove special characters and non-alphanumeric tokens
    temp = []
    for i in sms:
        if i.isalnum():
            temp.append(i)
            
    # Remove stopwords and punctuation marks
    sms = temp[:]
    temp.clear()
    for i in sms:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)
    sms = temp[:]
    temp.clear()

    # Stemming using PorterStemmer
    for i in sms:
        temp.append(ps.stem(i))
        
    # Return preprocessed SMS as a string
    return " ".join(temp)

# Predict button and result display
if st.button('Predict'):
    Usable_sms = usable_sms(Input_sms)
    vec_input = tfidf.transform([Usable_sms])

    result = preferred_model.predict(vec_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
