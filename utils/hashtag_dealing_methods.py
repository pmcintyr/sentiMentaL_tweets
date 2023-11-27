import re
import multiprocessing

from collections import deque

def split_hashtag(text, token='#', vocabulary=set()):
    # Find the word(s) following the token
    match = re.search(f"{token}(\w+)", text)
    if not match:
        return []

    hashtag = match.group(1)
    
    # Check if the entire hashtag is in the vocabulary
    full_hashtag = token + hashtag
    if full_hashtag in vocabulary:
        return [full_hashtag]
    
    
    words = []
    remaining = deque(hashtag)

    while remaining:
        current = ''
        best_match = None

        for char in list(remaining):
            current += char
            if current in vocabulary:
                # If current match is earlier in the vocabulary, update best_match
                if best_match is None or vocabulary.index(current) < vocabulary.index(best_match):
                    best_match = current

        if best_match:
            words.append(best_match)
            for _ in range(len(best_match)):
                remaining.popleft()  # Remove the matched part from remaining
        else:
            break  # No further matches found

    return words if words else [full_hashtag]  # Return the original hashtag if no words found



def remove_hashtags(train_tweets, test_tweets, vocabulary):
    # Remove '#' characters from each tweet in the list
    train_tweets = [tweet.replace("#", "") for tweet in train_tweets]
    test_tweets = [tweet.replace("#", "") for tweet in test_tweets]
    vocabulary = [word.replace("#", "") for word in vocabulary]
    
    return


# def process_tweets_hashtags(train_tweets_pos, train_tweets_neg, test_tweets, vocabulary, clean_data_again):
#     if clean_data_again:
#         processed_train_tweets_pos = []
#         hashtag_regex = re.compile(r"#\w+")

        
#         for tweet in train_tweets_pos:
#             hashtags = hashtag_regex.findall(tweet)
#             processed_tweet = tweet

#             for hashtag in hashtags:
#                 split_words = split_hashtag(hashtag, vocabulary=vocabulary)
#                 processed_tweet = processed_tweet.replace(hashtag, ' '.join(split_words))

#             processed_train_tweets_pos.append(processed_tweet)
            
#         processed_train_tweets_neg = []
        
#         for tweet in train_tweets_neg:
#             hashtags = hashtag_regex.findall(tweet)
#             processed_tweet = tweet

#             for hashtag in hashtags:
#                 split_words = split_hashtag(hashtag, vocabulary=vocabulary)
#                 processed_tweet = processed_tweet.replace(hashtag, ' '.join(split_words))

#             processed_train_tweets_neg.append(processed_tweet)
            
#         processed_test_tweets = []
        
#         for tweet in test_tweets:
#             hashtags = hashtag_regex.findall(tweet)
#             processed_tweet = tweet

#             for hashtag in hashtags:
#                 split_words = split_hashtag(hashtag, vocabulary=vocabulary)
#                 processed_tweet = processed_tweet.replace(hashtag, ' '.join(split_words))

#             processed_test_tweets.append(processed_tweet)
            
#         # Write the contents into a new file
#         with open('processed_data/train_pos.txt', 'w', encoding='utf-8') as file:
#             file.writelines(processed_train_tweets_pos)
            
#         with open('processed_data/train_neg.txt', 'w', encoding='utf-8') as file:
#             file.writelines(processed_train_tweets_neg)
            
#         with open('processed_data/test_data.txt', 'w', encoding='utf-8') as file:
#             file.writelines(processed_test_tweets)
                    
#     else:
        
#         # Load the processed data
#         with open('processed_data/train_pos.txt', 'r', encoding='utf-8') as file:
#             processed_train_tweets_pos = file.readlines()
            
#         with open('processed_data/train_neg.txt', 'r', encoding='utf-8') as file:
#             processed_train_tweets_neg = file.readlines()
            
#         with open('processed_data/test_data.txt', 'r', encoding='utf-8') as file:
#             processed_test_tweets = file.readlines()
        
    
#     return processed_train_tweets_pos, processed_train_tweets_neg, processed_test_tweets










def process_tweet(tweet, vocabulary):
    hashtag_regex = re.compile(r"#\w+")
    hashtags = hashtag_regex.findall(tweet)
    processed_tweet = tweet

    for hashtag in hashtags:
        split_words = split_hashtag(hashtag, vocabulary=vocabulary)
        processed_tweet = processed_tweet.replace(hashtag, ' '.join(split_words))

    return processed_tweet

def process_tweets_hashtags_parallel(tweets, vocabulary):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        processed_tweets = pool.starmap(process_tweet, [(tweet, vocabulary) for tweet in tweets])

    return processed_tweets

def process_tweets_hashtags(train_tweets_pos, train_tweets_neg, test_tweets, vocabulary, clean_data_again):
    if clean_data_again:
        processed_train_tweets_pos = process_tweets_hashtags_parallel(train_tweets_pos, vocabulary)
        processed_train_tweets_neg = process_tweets_hashtags_parallel(train_tweets_neg, vocabulary)
        processed_test_tweets = process_tweets_hashtags_parallel(test_tweets, vocabulary)

        # Write the contents into files
               # Write the contents into a new file
        with open('processed_data/train_pos.txt', 'w', encoding='utf-8') as file:
            file.writelines(processed_train_tweets_pos)
            
        with open('processed_data/train_neg.txt', 'w', encoding='utf-8') as file:
            file.writelines(processed_train_tweets_neg)
            
        with open('processed_data/test_data.txt', 'w', encoding='utf-8') as file:
            file.writelines(processed_test_tweets)

    else:
        # Load the processed data
        with open('processed_data/train_pos.txt', 'r', encoding='utf-8') as file:
            processed_train_tweets_pos = file.readlines()
            
        with open('processed_data/train_neg.txt', 'r', encoding='utf-8') as file:
            processed_train_tweets_neg = file.readlines()
            
        with open('processed_data/test_data.txt', 'r', encoding='utf-8') as file:
            processed_test_tweets = file.readlines()

    return processed_train_tweets_pos, processed_train_tweets_neg, processed_test_tweets

# Replace your existing call to process_tweets_hashtags with main_process
