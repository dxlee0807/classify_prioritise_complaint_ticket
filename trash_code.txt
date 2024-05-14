import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
# from nltk.util import ngrams
import string
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text Preprocessing Function for EDA
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # remove url
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

    # Remove punctuation
    # match all non-alphanumeric and non-whitespace characters,
    non_alpha_pattern = r"[^\w\s]"
    text = re.sub(non_alpha_pattern, " ", text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # POS tagging and lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, tag in pos_tag(tokens):
      wntag = tag[0].lower()
      wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
      lemmas.append(lemmatizer.lemmatize(word, wntag) if wntag else word)

    # furthwr cleaning
    # remove repeating characters from tokens
    RepeatTokensRm =  " ".join( [ re.sub(r'(\w)\1{2,}', r'\1', word) for word in lemmas] )
    # Remove tokens containing digits
    digitTokensRm =  " ".join( [ word for word in RepeatTokensRm.split() if not re.search(r'\d', word) ] ) 
    # Remove tokens containing underscore
    underscoreTokensRm =  " ".join( [ word for word in digitTokensRm.split() if not re.search(r'_|\w*_\w*', word) ] )
    # Remove tokens containing Special Characters
    specialTokensRm =  " ".join( [ word for word in underscoreTokensRm.split() if not re.search(r'[^a-zA-Z0-9\s]', word) ] )
    # Remove tokens less than 2 characters
    return " ".join( [ word for word in specialTokensRm.split() if len(word) > 2 ] )



# concat title and text
news_df["news"] = news_df["title"] + " " + news_df["text"] 

# drop unnecessary column
news_df.drop(labels = ["title",'text',"subject", "date"], axis = 1, inplace = True)

# rename news to text
news_df.rename(columns={"news": "text"},inplace=True)

# remove rows with only whitespaces
news_df['text'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
news_df.dropna(inplace=True)

# Apply text preprocessing to the 'text' column
news_df['cleaned_text'] = news_df['text'].apply(preprocess_text)
# news_df['unigrams'], news_df['bigrams'], news_df['trigrams'] = zip(*news_df['text'].apply(preprocess_text))

# Display the preprocessed text
news_df.head()




# unigram frequency
from nltk.tokenize import word_tokenize

text = " ".join(list(news_df['cleaned_text']))
uni_tokens = word_tokenize(text)
unigram_df = pd.DataFrame({'unigram':uni_tokens})

unigram_freq_df = unigram_df.groupby('unigram').size().reset_index(name='count').sort_values(by='count', ascending=False)
unigram_freq_df.iloc[:19]



d = {}
for a, x in unigram_freq_df.values:
    d[a] = x

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



# split into training set (test set?) and validation set

# train = 80, test = 20
# random_seed = 42

from sklearn.model_selection import train_test_split

X = clean_news_df['cleaned_text'].str.split()
y = clean_news_df['true_or_fake']

X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=True,random_state=42,test_size=0.2,stratify=y)

X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)



# train a Word2Vec model

from gensim.models import Word2Vec
w2v_model = Word2Vec(X_train, vector_size=200, window=5, min_count=1)

vocab=list(w2v_model.wv.key_to_index.keys())
print(len(vocab))


words = set(w2v_model.wv.index_to_key)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test])
X_train_avg = []
for v in X_train_vect:
        X_train_avg.append(v.mean(axis=0))

X_test_avg = []
for v in X_test_vect:
        X_test_avg.append(v.mean(axis=0))



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_avg, y_train)
lr.score(X_test_avg,y_test)




from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, classification_report

lr_preds = lr.predict(X_test_avg)
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=lr_preds))
print("Precision: ", precision_score(y_true=y_test, y_pred=lr_preds, pos_label='true'))
print("Recall: ", recall_score(y_true=y_test, y_pred=lr_preds, pos_label='true'))
print("F1-Score: ", f1_score(y_true=y_test, y_pred=lr_preds, pos_label='true'))
print(classification_report(y_test, lr_preds))

print("Confusion Matrix: \n", confusion_matrix(y_true=y_test, y_pred=lr_preds).ravel())




from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength parameter
    'solver': ['liblinear', 'saga']  # Solvers to consider
}
grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_avg, y_train)
print("Best hyperparameters:", grid_search.best_params_)

bestlr = grid_search.best_estimator_


best_lr_score = bestlr.score(X_test_avg, y_test)
print("Best model accuracy:", best_lr_score)

best_lr_preds = bestlr.predict(X_test_avg)


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, classification_report

print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=best_lr_preds))
print("Precision: ", precision_score(y_true=y_test, y_pred=best_lr_preds, pos_label='true'))
print("Recall: ", recall_score(y_true=y_test, y_pred=best_lr_preds, pos_label='true'))
print("F1-Score: ", f1_score(y_true=y_test, y_pred=best_lr_preds, pos_label='true'))
print(classification_report(y_test, best_lr_preds))

print("Confusion Matrix: \n", confusion_matrix(y_true=y_test, y_pred=best_lr_preds).ravel())