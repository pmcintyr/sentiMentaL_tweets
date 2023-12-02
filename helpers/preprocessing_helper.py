import re
import pkg_resources
import numpy as np
import pandas as pd
import nltk
import ssl
import sys

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import subprocess
import sys
from symspellpy import SymSpell

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

symspell = SymSpell()

dictionary_path = pkg_resources.resource_filename(
'symspellpy',
'frequency_dictionary_en_82_765.txt')

symspell.load_dictionary(dictionary_path, term_index=0,
                                        count_index=1)

bigram_path = pkg_resources.resource_filename(
'symspellpy',
'frequency_bigramdictionary_en_243_342.txt')

symspell.load_bigram_dictionary(bigram_path, term_index=0,
                                            count_index=2)

file_path = ['../twitter-datasets/train_neg.txt', '../twitter-datasets/train_pos.txt']
full_file_path = ['../twitter-datasets/train_neg_full.txt', '../twitter-datasets/train_pos_full.txt']
test_file_path = ['../twitter-datasets/test_data.txt']

def get_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def drop_duplicates():
    data = data.drop_duplicates(subset=['text'])
    
def remove_elongs():
    data['text'] = data['text'].apply(
      lambda text: str(re.sub(r'\b(\S*?)(.)\2{3,}\b', r'\1\2\2\2', text)))

def lower_case():
    data['text'] = data['text'].str.lower()

def spell_correct():
    data['text'] = data['text'].apply(lambda text: symspell.lookup_compound(text, max_edit_distance=2)[0].term)

def lemmatize(text):
    nltk_tagged = nltk.pos_tag(text.split())
    lemmatizer = WordNetLemmatizer()

    return ' '.join(
      [lemmatizer.lemmatize(w, get_wordnet_tag(nltk_tag))
       for w, nltk_tag in nltk_tagged])

def lemmatizer():
    data['text'] = data['text'].apply(lemmatize)

def stopword():
    stopwords_ = set(stopwords.words('english'))

    data['text'] = data['text'].apply(
      lambda text: ' '.join(
        [word for word in str(text).split() if word not in stopwords_]))
    
def hashtag():
    data['text'] = data['text'].apply(
      lambda text: str(re.sub(r'[\<].*?[\>]', '', text)))
    data['text'] = data['text'].apply(lambda text: text.strip())

def remove_tags():
    # data['text'] = data['text'].apply(
    #   lambda text: str(re.sub(r'[\<].*?[\>]', '', text)))
    data.replace(r'<.*?>', '', regex=True, inplace=True)
    data['text'] = data['text'].apply(lambda text: text.strip())
    data['text'] = data['text'].str.replace('\.{3}$', '')

def filter_alpha(tokens):
    return [word for word in tokens if word.isalpha()]

def letters():
    data['text'] = data['text'].apply(lambda text: filter_alpha(text.split()))

def prune_punctuations():
    data['text'] = data['text'].replace({'[$&+=@#|<>:*()%]': ''}, regex=True)

def empty():
    # data['text'] = data['text'].str.replace('^\s*$', '<EMPTY>')
    data.replace("", "<EMPTY>", inplace=True)

def spacing():
    # rewrite
    data['text'] = data['text'].str.replace('\s{2,}', ' ')
    data['text'] = data['text'].apply(lambda text: text.strip())
    data.reset_index(inplace=True, drop=True)

def nltk_resource():
    # Download the stopwords resource
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords')

def preprocess_tweet_old(tweet):
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
with open('../twitter-datasets/train_pos.txt', 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

with open('../twitter-datasets/train_neg.txt', 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

with open('../twitter-datasets/test_data.txt', 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()


processed_pos_tweets, processed_neg_tweets, processed_test_tweets = [], [], []

for pos_tweet, neg_tweet, test_tweet in zip(pos_tweets, neg_tweets, test_tweets):
    processed_pos_tweets.append(preprocess_tweet(pos_tweet))
    processed_neg_tweets.append(preprocess_tweet(neg_tweet))
    processed_test_tweets.append(preprocess_tweet(test_tweet))

# save to files
pos_file_path = '../twitter-datasets/processed_train_pos.txt'
neg_file_path = '../twitter-datasets/processed_train_neg.txt'
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

def main(argv):
    dataset = argv[0]
    model = argv[1]
    global data 
    data = pd.DataFrame(columns=['text', 'label'])

    if dataset == 'train':
        list = file_path
    elif dataset == 'train_full':
        list = full_file_path
    elif dataset == 'test':
        list = test_file_path
    else:
        list = ['../twitter-datasets/testing.txt']

    if dataset == 'test':
        with open(test_file_path[0]) as f:
            content = f.read().splitlines()
        ids = [line.split(',')[0] for line in content]
        texts = [','.join(line.split(',')[1:]) for line in content]
        data = pd.DataFrame(columns=['ids', 'text'],
                                    data={'ids': ids, 'text': texts})
    else:
        for i, path in enumerate(list):
            with open(path) as f:
                content = f.readlines()

                df = pd.DataFrame(columns=['text', 'label'],
                                data={'text': content,
                                    'label': np.ones(len(content)) * i})

                data = pd.concat([data, df], ignore_index=True)
        
    if dataset == 'train' or dataset == 'train_full':
        data = data.drop_duplicates(subset=['text'])

    if model == 'distilbert':
        lower_case()
        remove_tags()
        remove_elongs()
        prune_punctuations()
        spacing()
        empty()

    data = data.sample(frac=1)

    if dataset == 'train':
        data.to_csv('../twitter-datasets/processed_train.csv', index=False)
    elif dataset == 'train_full':
        data.to_csv('../twitter-datasets/processed_train_full.csv', index=False)
    elif dataset == 'test':
        data.to_csv('../twitter-datasets/processed_test.csv', index=False)
    else: print(data)

if __name__ == "__main__":
   print(sys.argv[1:])
   main(sys.argv[1:])