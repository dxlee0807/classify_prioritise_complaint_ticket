# expected file: 
# - tuned_lr.pkl, 
# - [topic_lda, topic_lda.id2word] to get topic information (actually need turn it into df and load)

# import packages
import os
from pickle import load
import pandas as pd
import json
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re
import contractions
import wordninja

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# load topic_csv
topic_df = pd.read_csv("lda_topic.csv")
# print(topic_df)

# load tfidf_vectorizer
with open(os.path.join(os.getcwd(),'ml_models',"vectorizer.pkl"), "rb") as f:
    vectorizer = load(f)

# load tuned_lr.pkl
with open(os.path.join(os.getcwd(),'ml_models',"tuned_lr.pkl"), "rb") as f:
    clf = load(f)


# define nlp pipeline
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

complaint_1 = """Unauthorized Charges on Credit Card
Customer Name: John Doe
Account Number: 1234 5678 9012 3456
Complaint Type: Fraudulent Charges
Description:
I recently noticed several unauthorized charges on my credit card account. The charges were made at various merchants and totaled $500. I did not make these charges and suspect my card may have been compromised. I would like the charges reversed and a new card issued immediately. Please investigate this matter and provide a resolution as soon as possible. I can be reached at 555-1234 or john.doe@email.com if you need any additional information."""

complaint_2 = """Mortgage Loan Modification Denial
Customer Name: Jane Smith
Loan Number: 98765-4321
Complaint Type: Loan Modification Denial
Description:
I am writing to appeal the denial of my mortgage loan modification request. I have been making reduced payments for the past 6 months due to a job loss, but have now secured a new position. However, my mortgage payments are still unaffordable. I believe I qualify for a modification under the terms of my loan agreement. Please review my file again and provide a decision within 30 days. I can provide additional income documentation if needed. I can be reached at 555-5678 or jane.smith@email.com."""

complaint_3 = """Debt Collection Harassment
Customer Name: Michael Johnson
Complaint Type: Debt Collection Harassment
Description:
I am being harassed by a debt collector regarding an old credit card debt. They have been calling me multiple times per day at home and work, despite me requesting they only contact me in writing. The calls are abusive and threatening. I have sent them a cease and desist letter, but the harassment continues. I would like this matter escalated to your legal department immediately. My attorney will be in touch if the calls do not stop. I can be reached at 555-9012 or michael.johnson@email.com."""

complaint_4 = """Unauthorized Bank Account Closure
Customer Name: Sarah Davis
Account Number: 54321-09876
Complaint Type: Unauthorized Account Closure
Description:
I visited my local bank branch today to make a deposit and was informed my checking account had been closed. No prior notice was provided. I rely on this account for my direct deposit paycheck and automatic bill payments. The branch manager was unable to reopen the account. I would like the account reinstated immediately with any related fees waived. Please investigate this matter and provide a resolution within 10 business days. I can be reached at 555-3456 or sarah.davis@email.com."""

complaint_5 = """Inaccurate Credit Report
Customer Name: David Wilson
Complaint Type: Inaccurate Credit Report
Description:
I recently obtained my credit report and noticed several inaccuracies. There are two credit card accounts listed that do not belong to me, along with a fraudulent mortgage inquiry. I have disputed this information directly with the credit bureaus, but the errors remain on my report. I would like these items removed immediately as they are negatively impacting my credit score. Please investigate this matter and provide a written response within 30 days. I can be reached at 555-7890 or david.wilson@email.com."""

def get_ticket_category_and_priority(text, topic_df, vectorizer, classifier):
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform(text.split(' '))
    pred = classifier.predict(text_tfidf)[0]

    # return pred[0], pred[1]
    category = topic_df.iloc[pred[0]][1]
    priority = pred[1]
    return category, priority

complaints = [complaint_1,complaint_2,complaint_3,complaint_4,complaint_5]

for c in complaints:
   print(get_ticket_category_and_priority(c, topic_df, vectorizer, clf))

# Financial Domain Complaint Ticketing System

# try using generative AI to generate some (3-5) financial complaint tickets

# enter title [short_text]

# enter description [long_text]

# preprocess the description
# - nlp, clean, tfidf
# - model.predict() to get ticket category, and priority


# can select which support user/team
# table (sortable and filterable)
# Description | Category | Priority | Resolve (tick to clear)


