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
    # data['text'] = data['text'].str.replace('\.{3}$', '')

def filter_alpha(tokens):
    return [word for word in tokens if word.isalpha()]

def letters():
    data['text'] = data['text'].apply(lambda text: filter_alpha(text.split()))

def empty():
    data['text'] = data['text'].str.replace('^\s*$', '<EMPTY>')

def main(argv):
    dataset = argv[0]
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

    for i, path in enumerate(list):
        with open(path) as f:
            content = f.readlines()

            df = pd.DataFrame(columns=['text', 'label'],
                            data={'text': content,
                                  'label': np.ones(len(content)) * i})

            data = pd.concat([data, df], ignore_index=True)
    
    data = data.drop_duplicates(subset=['text'])
    print(len(data))
    hashtag()
    remove_elongs()
    spell_correct()
    lemmatizer()
    stopword()
    empty()

    data = data.sample(frac=1)
    data.to_csv('../twitter-datasets/processed_train.csv', index=False)

    # Save each column as a text file
    for column in data.columns:
        # Construct the file name (you can customize this as needed)
        file_name = f"../twitter-datasets/{column}.txt"

        # Save the column to a text file
        data[column].to_csv(file_name, index=False, header=False)

if __name__ == "__main__":
   print(sys.argv[1:])
   main(sys.argv[1:])