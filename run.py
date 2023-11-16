### LOADING EMDEDDINGS AND DATA

import numpy as np

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

# Construct feature representations for training tweets
def average_word_vectors(tweet, embeddings, word_to_embedding):
    words = tweet.split()
    vectors = [word_to_embedding[word] for word in words if word in word_to_embedding]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(embeddings[0])

# Construct feature representations for training tweets
train_features = [average_word_vectors(tweet, word_embeddings, word_to_embedding) for tweet in train_tweets]

### TRAINING THE LINEAR CLASSIFIER

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Construct Features for Training Texts

def average_word_vectors(tweet, word_to_embedding):
    words = tweet.split()
    vectors = [word_to_embedding[word] for word in words if word in word_to_embedding]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_to_embedding["you"])


# Construct feature representations for training tweets
train_features = [average_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]

# Step 3: Train a Linear Classifier

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, labels, test_size=0.1, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")


# Construct feature representations for test tweets
test_features = [average_word_vectors(tweet, word_embeddings) for tweet in test_tweets]

# Make predictions
y_test_pred = model.predict(test_features)

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
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

ids_test = get_test_ids(test_data_path)
y_pred = []
y_pred = y_test_pred
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
create_csv_submission(ids_test, y_pred, "submission")