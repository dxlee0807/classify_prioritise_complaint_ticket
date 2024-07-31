# the backend of the model deployment
# - text preprocessing
# - text representation
# - predict ticket category and ticket priority

# import packages
import os
from pickle import load
import pandas as pd

import nltk
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re
import contractions
import wordninja

def load_files():
    # load ticket_category CSV file
    topic_df = pd.read_csv("ticket_category.csv")

    # load tfidf_vectorizer
    with open(os.path.join(os.getcwd(),'ml_models',"vectorizer.pkl"), "rb") as f:
        vectorizer = load(f)

    # load tuned_lr.pkl
    with open(os.path.join(os.getcwd(),'ml_models',"lr_tfidf_20240727_1.pkl"), "rb") as f:
        classifier = load(f)

    return topic_df, vectorizer, classifier

# define text preprocessing pipeline
def clean_text(text):
  # Case-folding (Lowercase)
  text = text.lower()
  
  # Split Concatenated Words
  text = " ".join(wordninja.split(text))

  # Remove url
  url_pattern = re.compile(r'(https?://\S+)|(www\.\S+)|(\S+\.\S+/\S+)')
  text = url_pattern.sub(r'', text)

  # Remove emoji
  emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"  # other miscellaneous symbols
                                u"\U000024C2-\U0001F251"  # enclosed characters
                              "]+", flags=re.UNICODE)
  text = emoji_pattern.sub(r'', text)

  # Expand Contractions
  text = contractions.fix(text)
  
  # Remove Punctuation, and words containing numbers
  punt_pattern = '[^\w\s]'
  word_with_num_pattern = '\w*\d\w*'
  text = re.sub(punt_pattern, '', text)
  text = re.sub(word_with_num_pattern, '', text)

  # Tokenisation
  tokens = word_tokenize(text)

  # Remove stopwords
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]

  return tokens

def lemmatise_with_pos_tagged(tokens):
  lemmatizer = WordNetLemmatizer()
  lemmas = []
  for word, tag in pos_tag(tokens):
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    lemmas.append(lemmatizer.lemmatize(word, wntag) if wntag else word)
  return lemmas

def further_clean(tokens):
  # remove repeating characters from tokens
  RepeatTokensRm =  " ".join( [ re.sub(r'(\w)\1{2,}', r'\1', word) for word in tokens] )
  # Remove tokens containing digits
  digitTokensRm =  " ".join( [ word for word in RepeatTokensRm.split() if not re.search(r'\d', word) ] ) 
  # Remove tokens containing underscore
  underscoreTokensRm =  " ".join( [ word for word in digitTokensRm.split() if not re.search(r'_|\w*_\w*', word) ] )
  # Remove tokens containing Special Characters
  specialTokensRm =  " ".join( [ word for word in underscoreTokensRm.split() if not re.search(r'[^a-zA-Z0-9\s]', word) ] )
  # Remove tokens less than 2 characters
  return " ".join( [ word for word in specialTokensRm.split() if len(word) > 2 ] )

def preprocess_text(text):
    tokens = clean_text(text)
    lemmas = lemmatise_with_pos_tagged(tokens)
    preprocessed_text = further_clean(lemmas)
    return preprocessed_text

def get_ticket_category_and_priority(text, topic_df, vectorizer, classifier):
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform(text.split(' '))
    pred = classifier.predict(text_tfidf)[0]
    category = topic_df.iloc[pred[0]][1]
    priority = pred[1]
    return category, priority
