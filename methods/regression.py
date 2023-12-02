use_google_colab = False


#set if we want to clean the or load the precleaned data
clean_data_again = True
# set a debug mode
debug = True

if debug and clean_data_again:
  #clean_data_again = False
  print("Warning: debug mode is on and clean_data_again has been reset to False.")

if use_google_colab :
  from google.colab import drive
  drive.mount('/content/drive')
  %cd /content/drive/MyDrive/ColabNotebooks/sentiMentaL_tweets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

import sys
import os

# Get the current working directory
current_dir = os.getcwd()

# Adjust the path to point to the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)


import utils_for_regression.vocab_manip as vm
import utils_for_regression.hashtag_dealing_methods as hdm
import utils_for_regression.pooling as po
import utils_for_regression.submission as sub

### DATA LOADING
# Load the word embeddings
word_embeddings = np.load('../embeddings.npy')
df_word_embeddings = pd.DataFrame(word_embeddings)



# Load the test set tweets
with open('../twitter-datasets/test_data.txt', 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()
    df_test_tweets = pd.DataFrame(test_tweets)


# Load the vocabulary
with open('../vocab_cut.txt', 'r', encoding='utf-8') as file:
    vocabulary = file.read().splitlines()


# Create a dictionary to map words to their corresponding embeddings
word_to_embedding = {word: word_embeddings[i] for i, word in enumerate(vocabulary)}
df_word_embeddings = pd.DataFrame(word_to_embedding)



# Load positive training tweets and assign labels
with open('../twitter-datasets/train_pos_full.txt', 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

pos_labels = np.ones(len(pos_tweets), dtype=int)  # Assign label 1 for positive tweets
df_pos_tweets = pd.DataFrame(pos_tweets)



# Load negative training tweets and assign labels
with open('../twitter-datasets/train_neg_full.txt', 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

neg_labels = -1 * np.ones(len(neg_tweets), dtype=int)  # Assign label -1 for negative tweets
df_neg_tweets = pd.DataFrame(neg_tweets)



# Combine positive and negative tweets and labels
df_pos_tweets = pd.DataFrame(pos_tweets)
df_neg_tweets = pd.DataFrame(neg_tweets)



if debug:
    pos_tweets = pos_tweets[:10]
    neg_tweets = neg_tweets[:10]
    test_tweets = test_tweets[:10]
    #all_tweets = np.concatenate((train_tweets, test_tweets), axis=0)
    pos_labels = pos_labels[:10]
    neg_labels = neg_labels[:10]
    vocabulary = vocabulary[:100]


#reorder the vocabulary and the word embeddings according to the largest number of occurences first
vocabulary, word_embeddings = vm.reorder_vocabulary(pos_tweets, neg_tweets, test_tweets, vocabulary, word_embeddings, clean_data_again, save_counts=True)

# remove hashtags that are not in the vocabulary
pos_tweets, neg_tweets, test_tweets = hdm.process_tweets_hashtags(pos_tweets, neg_tweets, test_tweets, vocabulary, clean_data_again)

#save the words that are not in the vocabulary
vm.out_of_vocab_file(pos_tweets, neg_tweets, test_tweets, vocabulary, clean_data_again)


### TRAINING THE LINEAR CLASSIFIER

pooling_method = "weigth" # "mean", "max", "tfidf", "weigth"

train_features, test_features = po.get_features(pooling_method, pos_tweets, neg_tweets, test_tweets, word_to_embedding, vocabulary, clean_data_again)
# Split the data into training and validation sets
labels = np.concatenate((pos_labels, neg_labels), axis=0)


train_features = np.array(train_features)
labels = np.array(labels)
# Assuming train_features and labels are NumPy arrays
assert len(train_features) == len(labels), "Features and labels must be of the same length"

# Generate a permutation of indices
shuffled_indices = np.random.permutation(len(train_features))

# Apply the shuffled indices to both features and labels
shuffled_features = train_features[shuffled_indices]
shuffled_labels = labels[shuffled_indices]



X_train, X_val, y_train, y_val = train_test_split(train_features, labels, test_size=0.1, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)


### LINEAR CLASSIFIER PREDICTIONS

# Construct feature representations for test tweets

# Make predictions
y_test_pred = model.predict(test_features)

test_data_path = "../twitter-datasets/test_data.txt"
ids_test = sub.get_test_ids(test_data_path)
y_pred = []
y_pred = y_test_pred
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
sub.create_csv_submission(ids_test, y_pred, "../submissions/submission_"+pooling_method+"_pooling_and_regression.csv")

