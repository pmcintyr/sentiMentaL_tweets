import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import utils.vocab_manip as vm


### DEFINE FUNCTIONS

def average_word_vectors(tweet, word_to_embedding):
    # tweet is the sentence
    # words are the words in the sentence
    # word_to_embedding is the dictionary mapping words to their embeddings
    words = tweet.split()
    vectors = []
    for word in words:
        if word in word_to_embedding:
            vectors.append(word_to_embedding[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_embeddings[0])

def max_word_vectors(tweet, word_to_embedding):
    # tweet is the sentence
    # words are the words in the sentence
    # word_to_embedding is the dictionary mapping words to their embeddings
    words = tweet.split()
    vectors = []
    for word in words:
        if word in word_to_embedding:
            vectors.append(word_to_embedding[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_embeddings[0])


def weighted_average_word_vectors(tweet, word_to_embedding, weight_, vocabulary):
    words = tweet.split()
    vectors = []
    weights = []

    for word in words:
        if word in word_to_embedding:
            vectors.append(word_to_embedding[word])
        else:
            vectors.append(np.zeros_like(next(iter(word_to_embedding.values()))))

    for word in words:
        if word in vocabulary:
            word_idx = vocabulary.index(word)
            weights.append(weight_[word_idx])
        else:
            weights.append(0)

    weights = np.array(weights)


    if np.any(weights):  # Check if all weights are not zero
        weights /= np.sum(weights)
    else:
        weights = np.ones_like(weights) / len(weights)

    return np.average(vectors, axis=0, weights=weights) # Return the weighted average


def get_features(pooling_method, train_tweets_pos, train_tweets_neg, test_tweets, word_to_embedding, vocabulary, clean_data_again):
    if pooling_method == "mean" :
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg), axis=0)
        # Construct feature representations for training tweets
        train_features = [average_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]
        test_features = [average_word_vectors(tweet, word_to_embedding) for tweet in test_tweets]
        
    elif pooling_method == "max" :
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg))

        train_features = [max_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]
        test_features = [max_word_vectors(tweet, word_to_embedding) for tweet in test_tweets]

    elif pooling_method == "tfidf":
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg))

        all_features = []
        all_tweets = np.concatenate((train_tweets, test_tweets))
        
        # Create the vectorizer
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)

        # Fit and transform the tweets
        tfidf = vectorizer.fit_transform(all_tweets)

        for doc_index, tweet in enumerate(all_tweets):
            tfidf_vector = tfidf[doc_index].todense().A1  # Convert to dense format and flatten
            feature = weighted_average_word_vectors(tweet, word_to_embedding, tfidf_vector, vocabulary)
            all_features.append(feature)

        train_features = all_features[:len(train_tweets)]
        test_features = all_features[len(train_tweets):]
        
    elif pooling_method == "weigth":
        weights = vm.calculate_weights(train_tweets_pos, train_tweets_neg, vocabulary, clean_data_again)
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg))
        all_features = []
        all_tweets = np.concatenate((train_tweets, test_tweets))
        
        for _, tweet in enumerate(all_tweets):
            feature = weighted_average_word_vectors(tweet, word_to_embedding, weights, vocabulary)
            all_features.append(feature)
            
        train_features = all_features[:len(train_tweets)]
        test_features = all_features[len(train_tweets):]
        
            
    else : 
        raise ValueError("Pooling method not recognized")
            
    return train_features, test_features