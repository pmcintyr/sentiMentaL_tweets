import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove mentions
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#', '', tweet)
    
    # Tokenization using NLTK's TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    
    # Remove punctuation and numbers
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming using NLTK's PorterStemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Reassemble the tokens into a processed tweet
    processed_tweet = ' '.join(tokens)
    
    return processed_tweet

# Example usage
# tweet = "Great tutorial on #SentimentAnalysis using NLTK! @username http://example.com #NLP"
# processed_tweet = preprocess_tweet(tweet)
# print("Original tweet:", tweet)
# print("Processed tweet:", processed_tweet)

# Load positive training tweets and assign labels
with open('../twitter-datasets/train_pos_full.txt', 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

with open('../twitter-datasets/train_neg_full.txt', 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

with open('../twitter-datasets/test_data.txt', 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()


processed_pos_tweets, processed_neg_tweets, processed_test_tweets = [], [], []

for pos_tweet, neg_tweet, test_tweet in zip(pos_tweets, neg_tweets, test_tweets):
    processed_pos_tweets.append(preprocess_tweet(pos_tweet))
    processed_neg_tweets.append(preprocess_tweet(neg_tweet))
    processed_test_tweets.append(preprocess_tweet(test_tweet))

# save to files
pos_file_path = '../twitter-datasets/processed_train_pos_full.txt'
neg_file_path = '../twitter-datasets/processed_train_neg_full.txt'
test_file_path = '../twitter-datasets/processed_test_data.txt'

with open(pos_file_path, 'w', encoding='utf-8') as file:
    for tweet in processed_pos_tweets:
        file.write(tweet + '\n')

with open(neg_file_path, 'w', encoding='utf-8') as file:
    for tweet in processed_neg_tweets:
        file.write(tweet + '\n')

with open(test_file_path, 'w', encoding='utf-8') as file:
    for tweet in processed_test_tweets:
        file.write(tweet + '\n')