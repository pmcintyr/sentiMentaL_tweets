import numpy as np
import torch
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_test_ids(path):
    file = open(path,'r')
    lines = file.readlines()
    for rowidx in range(len(lines)):
        index = lines[rowidx].index(',')
        lines[rowidx] = lines[rowidx][:index]
    return lines

def create_csv_submission(ids, y_pred, name):
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

test_data_path = 'twitter-datasets/processed_test_data.txt'
train_pos_path = 'twitter-datasets/processed_train_pos.txt'
train_neg_path = 'twitter-datasets/processed_train_neg.txt'

# Load the test set tweets
with open(test_data_path, 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()

# Load positive training tweets and assign labels
with open(train_pos_path, 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

pos_labels = np.ones(len(pos_tweets), dtype=int)  # Assign label 1 for positive tweets

# Load negative training tweets and assign labels
with open(train_neg_path, 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

neg_labels = 0 * np.ones(len(neg_tweets), dtype=int)  # Assign label -1 for negative tweets

tweets = pos_tweets + neg_tweets
labels = np.concatenate((pos_labels, neg_labels), axis=0)

# Split the data into training and testing sets
train_tweets, test_tweets, train_labels, test_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)

# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # Assuming binary classification

# Tokenize and encode tweets for training set   
train_input_ids = [tokenizer.encode(tweet, add_special_tokens=True, max_length=512) for tweet in train_tweets]
train_input_ids = torch.tensor(pad_sequences(train_input_ids, padding='post', truncating='post', maxlen=None))
# train_labels = torch.tensor(train_labels)

# Tokenize and encode tweets for testing set
test_input_ids = [tokenizer.encode(tweet, add_special_tokens=True, max_length=512) for tweet in test_tweets]
test_input_ids = torch.tensor(pad_sequences(test_input_ids, padding='post', truncating='post', maxlen=None))
test_labels = torch.tensor(test_labels)

# # Tokenize and pad sequences
# input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=512) for sent in tokenized_data]
# input_ids = torch.tensor(pad_sequences(input_ids, padding='post', truncating='post', maxlen=None))

# Create a DataLoader
dataset = TensorDataset(train_input_ids, torch.tensor(train_labels))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

print("start fine-tune")

# Fine-tune the model
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    print(epoch)
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, labels=labels)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()

## VALIDATION & PREDICTIONS ###

# Validate
test_dataset = TensorDataset(test_input_ids, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Lists to store predictions
all_predictions = []

# Iterate through the test dataloader
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass to get logits
        logits = model(inputs)

        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get the predicted class (index with the maximum probability)
        _, predicted_class = torch.max(probabilities, 1)

        # Append predictions to the list
        all_predictions.extend(predicted_class.cpu().numpy())

# Convert the list to a numpy array
all_predictions = np.array(all_predictions)

# Calculate accuracy
accuracy = accuracy_score(test_labels.cpu().numpy(), all_predictions)
print(f"Validation Accuracy: {accuracy}")

# Make predictions
y_test_pred = all_predictions

#-----------------------------------------------------------------------------------------------------#
### CREATE CSV SUBMISSION ###

ids_test = get_test_ids('twitter-datasets/test_data.txt')
y_pred = []
y_pred = y_test_pred
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
create_csv_submission(ids_test, y_pred, "submission_dense.csv")