import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

'''
    Method to preprocess text data by removing special characters, converting to lowercase, 
    removing stopwords, and lemmatizing and tokenizing the words. All pretty standard NLP
    preprocessing steps. 
'''

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[,.!?:()"]', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text)
    
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)
