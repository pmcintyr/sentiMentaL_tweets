### IMPORTS AND CONSTANTS

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
# !pip install scikit-learn

test_data_path = 'twitter-datasets/test_data.txt'

### DATA LOADING

# Load the word embeddings
word_embeddings = np.load('embeddings.npy')

# Load the test set tweets
with open('twitter-datasets/test_data.txt', 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()

# Load the vocabulary
with open('vocab_cut.txt', 'r', encoding='utf-8') as file:
    vocabulary = file.read().splitlines()

# Create a dictionary to map words to their corresponding embeddings
word_to_embedding = {word: word_embeddings[i] for i, word in enumerate(vocabulary)}

# Load positive training tweets and assign labels
with open('twitter-datasets/train_pos_full.txt', 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

pos_labels = np.ones(len(pos_tweets), dtype=int)  # Assign label 1 for positive tweets

# Load negative training tweets and assign labels
with open('twitter-datasets/train_neg_full.txt', 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

neg_labels = -1 * np.ones(len(neg_tweets), dtype=int)  # Assign label -1 for negative tweets

# Combine positive and negative tweets and labels
train_tweets = pos_tweets + neg_tweets
labels = np.concatenate((pos_labels, neg_labels), axis=0)

### DEFINE FUNCTIONS

def average_word_vectors(tweet, word_to_embedding):
    words = tweet.split()
    vectors = [word_to_embedding[word] for word in words if word in word_to_embedding]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_embeddings[0])
    
def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def get_test_ids(path):
    file = open(path,'r')
    lines = file.readlines()
    for rowidx in range(len(lines)):
        index = lines[rowidx].index(',')
        lines[rowidx] = lines[rowidx][:index]
    return lines

# Construct feature representations for training and testing tweets
train_features = [average_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]
test_features = [average_word_vectors(tweet, word_to_embedding) for tweet in test_tweets]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, labels, test_size=0.1, random_state=42)

#-----------------------------------------------------------------------------------------------------#
####### MODELS #######

### TRAINING LOGISTIC REGRESSION
# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

### TRAINING SVM
# Standardize features (important for SGD)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_val)

# Create an SGDClassifier with a linear SVM loss
svm_classifier = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=100, random_state=42, learning_rate='optimal', eta0=0.0, early_stopping=True, n_iter_no_change=5)
model.fit(X_train, y_train)

#-----------------------------------------------------------------------------------------------------#
### VALIDATION & PREDICTIONS ###

# Validate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

### LOGISTIC REGRESSION PREDICTIONS
# Make predictions
y_test_pred = model.predict(test_features)
print(y_test_pred)

#-----------------------------------------------------------------------------------------------------#
### CREATE CSV SUBMISSION ###

ids_test = get_test_ids(test_data_path)
y_pred = []
y_pred = y_test_pred
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
create_csv_submission(ids_test, y_pred, "submission_svm.csv")