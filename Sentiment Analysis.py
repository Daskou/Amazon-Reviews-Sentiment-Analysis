"""Sentiment Analysis Machine Learning Project
Daskalakis Alexandros
ID : 0011
email : math1p0011@math.uoc.gr

January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
import string
from nltk.stem import WordNetLemmatizer
from pandarallel import pandarallel
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score ,  accuracy_score
import re

#Data cleaning and data preparation
df = pd.DataFrame()

# Read the JSONL files in chunks and combine them

chunksize = 100000
files = ['Health_and_Personal_Care.jsonl','Magazine_Subscriptions.jsonl']

# Use tqdm to track the progress
for file in files:
    # Get the total number of chunks for the current file
    total_chunks = sum(1 for _ in open(file)) // chunksize + 1
    with tqdm(total=total_chunks, desc=f'Processing {file}') as pbar:
        for chunk in pd.read_json(file, lines=True, chunksize=chunksize):
            chunk['source_file'] = file
            df = pd.concat([df, chunk], axis=0, ignore_index=True)
            pbar.update(1)  # Update the progress bar for each chunk processed


df= df.drop(columns=['images', 'asin', 'parent_asin', 'user_id',
     'timestamp', 'helpful_vote', 'verified_purchase', 'source_file', 'title'])

df = df.dropna(subset=['text'])


df = df[df['text'].str.strip() != '']  # Remove rows with empty or whitespace-only text
df['text'] = df['text'].astype(str)


df = df[df['text'].apply(lambda x: isinstance(x, str))]  # Keep only string entries


def assign_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating'].apply(assign_sentiment)


lemmatizer = WordNetLemmatizer()

stopwords = set(stopwords.words('english'))

stop_punc_list = stopwords.union(string.punctuation)

df = df.drop(columns=['rating'])

X = df['text'].sample(n=10000, random_state=42)
X = pd.DataFrame(X)

Y = df.loc[X.index, 'sentiment']  # Get the labels corresponding to the sampled rows
y = pd.DataFrame(Y)


def preprocessing(df):
    processed_texts = []  # Store processed text
    original_texts = []   # Store original text
    text_ids = []         # Store text indices

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        text = row['text']
        text_id = i  # Store text ID
        
        # Tokenization and stopword removal
        tokenized_text = nltk.word_tokenize(text)
        filtered_text = [
            token for token in tokenized_text 
            if token not in stop_punc_list           # Remove stopwords & punctuation
            and not re.search(r'\d', token)         # Remove numbers
            and re.match(r'^[a-zA-Z]+$', token)    # Keep only alphabetic words
        ]     
        final_text = " ".join(filtered_text)
        
        processed_texts.append(final_text)
        original_texts.append(text)
        text_ids.append(text_id)

    # Return preprocessed text as DataFrame
    return pd.DataFrame({"Text_ID": text_ids, "Processed Text": processed_texts})





#Cross Validation
#Tuning hyper-parameters
alphas = [0.05 , 0.5 , 0.025]
Cs = [1, 1.5, 2]  # Regularization strengths for Logistic Regression

#Models used
classifiers = {
    "Naive Bayes": lambda alpha: MultinomialNB(alpha=alpha),
    "Logistic Regression": lambda C: LogisticRegression(C=C, max_iter=1000)
}

kf = KFold(n_splits=5, shuffle=True) #k=5
best_model = None
best_score = -1  # Initialize best score with a low value


for clf_name, clf_constructor in classifiers.items():
    param_list = alphas if clf_name == "Naive Bayes" else Cs
    
    for param in param_list:
        print(f"Training {clf_name} with parameter={param}")
        model = clf_constructor(param)
        fold_accuracies = []
        fold_f1_scores = []  # List to store F1-scores for each fold

        for train_index, val_index in kf.split(X):

        #Splitting into X_train and x_validation
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]  
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
            y_train = y_train.values.ravel()
            y_val = y_val.values.ravel()      

        
            #preprocess X_train in order to create the TF-IDF Values.
            train_df = preprocessing(X_train)
            vectorizer = TfidfVectorizer(max_features=5000) # we set how many.
            X_train = vectorizer.fit_transform(train_df["Processed Text"])
            X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
            X_train.insert(1, "Text_ID", train_df["Text_ID"])  # Keep track of original text index
            X_train.set_index('Text_ID', inplace=True)
    
            
            #Preprocess Test Data
            test_df = preprocessing(X_val)
    
            #Transform X_test using the same vectorizer
            X_val= vectorizer.transform(test_df["Processed Text"])
            X_val = pd.DataFrame(X_val.toarray(), columns=vectorizer.get_feature_names_out())
            X_val.insert(1,"Text_ID" , test_df["Text_ID"])
            X_val.set_index('Text_ID', inplace=True)

        
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')  # Calculate weighted F1-score
            fold_f1_scores.append(f1)
            fold_accuracies.append(accuracy)

        # Compute Average Accuracy for the Parameter
        avg_accuracy = np.mean(fold_accuracies)
        avg_f1_score = np.mean(fold_f1_scores)

        print(f"Average accuracy for {clf_name} with parameter={param}: {avg_accuracy}")
        print(f"Average F1-score for {clf_name} with parameter={param}: {avg_f1_score}")

        # Update Best Model if Current Model Performs Better
        if avg_f1_score > best_score:
            best_score = avg_f1_score
            best_model = model
            best_classifier = f"{clf_name} (param={param})"
            best_acc = avg_accuracy

# Print Best Model Found
print(f"Best model: {best_classifier} with F1-Score={best_score} and accuracy={best_acc}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Preprocess Training Data
train_df = preprocessing(X_train)

#Fit TF-IDF on the training set
vectorizer = TfidfVectorizer(max_features=5000) # we set how many.
X_train = vectorizer.fit_transform(train_df["Processed Text"])
X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
X_train.insert(1, "Text_ID", train_df["Text_ID"])  # Keep track of original text index
X_train.set_index('Text_ID', inplace=True)

#Preprocess Test Data
test_df = preprocessing(X_test)

#Transform X_test using the same vectorizer
X_test= vectorizer.transform(test_df["Processed Text"])
X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
X_test.insert(1,"Text_ID" , test_df["Text_ID"])
X_test.set_index('Text_ID', inplace=True)



y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


#Fitting the model one last time without CV
best = LogisticRegression(C=2)

best.fit(X_train,y_train)

y_pred = best.predict(X_test)


accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate weighted F1-score
print(f"F1-Score for Best Model: {f1:.4f}")

print(f"The Best Model has final accuracy: {accuracy:.4f}")