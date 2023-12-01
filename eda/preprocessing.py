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
    data['text'] = data['text'].str.replace('<[\w]*>', '')
    data['text'] = data['text'].apply(lambda text: text.strip())
    data['text'] = data['text'].str.replace('\.{3}$', '')

def filter_alpha(tokens):
    return [word for word in tokens if word.isalpha()]

def letters():
    data['text'] = data['text'].apply(lambda text: filter_alpha(text.split()))

def prune_punctuations():
    data['text'] = data['text'].replace({'[$&+=@#|<>:*()%]': ''}, regex=True)

def empty():
    data['text'] = data['text'].str.replace('^\s*$', '<EMPTY>')

def spacing():
    # rewrite
    data['text'] = data['text'].str.replace('\s{2,}', ' ')
    data['text'] = data['text'].apply(lambda text: text.strip())
    data.reset_index(inplace=True, drop=True)

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

    data = data.sample(frac=1)
    if dataset == 'train':
        data.to_csv('../twitter-datasets/processed_train.csv', index=False)
    elif dataset == 'train_full':
        data.to_csv('../twitter-datasets/processed_train_full.csv', index=False)
    else:
        data.to_csv('../twitter-datasets/processed_test.csv', index=False)

    # Save each column as a text file
    # for column in data.columns:
    #     # Construct the file name (you can customize this as needed)
    #     file_name = f"../twitter-datasets/{column}.txt"

    #     # Save the column to a text file
    #     data[column].to_csv(file_name, index=False, header=False)

if __name__ == "__main__":
   main(sys.argv[1:])