# expected file: 
# - tuned_lr.pkl, 
# - [topic_lda, topic_lda.id2word] to get topic information (actually need turn it into df and load)

# import packages
import os
from pickle import load
import datetime
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

import altair as alt
import streamlit as st

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

# preprocess the description
# - nlp, clean, tfidf
# - model.predict() to get ticket category, and priority
def get_ticket_category_and_priority(text, topic_df, vectorizer, classifier):
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform(text.split(' '))
    pred = classifier.predict(text_tfidf)[0]
    category = topic_df.iloc[pred[0]][1]
    priority = pred[1]
    return category, priority

# complaints = [complaint_1,complaint_2,complaint_3,complaint_4,complaint_5]

# for c in complaints:
#    print(get_ticket_category_and_priority(c, topic_df, vectorizer, clf))

# Financial Domain Complaint Ticketing System
st.set_page_config(page_title="Financial Domain Complaint Ticketing System", page_icon="🎫")
st.title("🎫 Financial Domain Complaint Ticketing System")
st.write(
    """
    This app shows how you can build an internal tool in Streamlit. Here, we are 
    implementing a support ticket workflow. The user can create a ticket, edit 
    existing tickets, and view some statistics.
    """
)

# Create a Pandas dataframe to store tickets.
if "ticket_df" not in st.session_state:
   data = {
      "ID": [], # TICKET-{id_in_int}
      "Title":[],
      "Status":[],
      "Description":[],
      "Category":[],
      "Priority":[],
      "Date Submitted":[]
   }
   ticket_df = pd.DataFrame(data)
   # Save the dataframe in session state (a dictionary-like object that persists across
   # page runs). This ensures our data is persisted when the app updates.
   st.session_state.ticket_df = ticket_df

# Show a section to add a new ticket.
st.header("Add a ticket")

# We're adding tickets via an `st.form` and some input widgets. If widgets are used
# in a form, the app will only rerun once the submit button is pressed.
with st.form("add_ticket_form"):
    title = st.text_input("Title")
    issue = st.text_area("Describe the issue") # enter description [long_text]
    submitted = st.form_submit_button("Submit")

if submitted:
   # Make a dataframe for the new ticket and append it to the dataframe in session state.
    # if int_x == 0 then recent_ticket_number
    try:
       recent_ticket_number = int(max(st.session_state.ticket_df.ID).split("-")[1])
    except ValueError:
       recent_ticket_number = 0
    category,priority = get_ticket_category_and_priority(issue, topic_df, vectorizer, clf)
    today = datetime.datetime.now().strftime("%Y-%d-%m, %H:%M:%S")
    # df add new row at the top of current ticket_df
    df_new = pd.DataFrame(
       [
          {
            "ID": f"TICKET-{recent_ticket_number+1}", # TICKET-{id_in_int}
            "Title":title,
            "Status":"Open",
            "Description":issue,
            "Category":category,
            "Priority":priority,
            "Date Submitted":today
          }
       ]
    )
    # Show a little success message.
    # show a box rather than df
    st.write("Ticket submitted! Here are the ticket details:")
    st.dataframe(df_new, use_container_width=True, hide_index=True)
    st.session_state.ticket_df = pd.concat([df_new, st.session_state.ticket_df], axis=0)

# can select which support user/team
# table (sortable and filterable)
# Description | Category | Priority | Status (tick to clear)

# Show section to view and edit existing tickets in a table.
st.header("Existing tickets")
st.write(f"Number of tickets: `{len(st.session_state.ticket_df)}`")

st.info(
    "You can edit the tickets by double clicking on a cell. Note how the plots below "
    "update automatically! You can also sort the table by clicking on the column headers.",
    icon="✍️",
)

# Show the tickets dataframe with `st.data_editor`. This lets the user edit the table
# cells. The edited data is returned as a new dataframe.
edited_df = st.data_editor(
    st.session_state.ticket_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn(
            "Status",
            help="Ticket status",
            options=["Open", "In Progress", "Closed"],
            required=True,
        )
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=["ID", "Title","Description","Category", "Priority","Date Submitted"],
)