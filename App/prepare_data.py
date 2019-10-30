import re
import json
from tqdm import tqdm

def decontracted(phrase):
    # specific
    phrase = re.sub(r"[Ww]on\'t", "will not", phrase)
    phrase = re.sub(r"[Cc]an\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    # phrase = re.sub(r"\'s", " is", phrase)
    # phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def split_review(phrase):
    phrases = phrase.split('\n')
    phrases = list(filter(None, phrases))
    return phrases

def merge_phrases(phrases):
    phrases = " ".join(phrases)
    return re.sub(r"  ", " ", phrases)


review_dic = {}
data_file = '../Data/reviewSelected100.json'

with open(data_file, 'r') as json_file:
    reviews = json_file.readlines()

for i in tqdm(range(len(reviews))):
    review = json.loads(reviews[i])
    review['text'] = merge_phrases(split_review(decontracted(review['text'])))
    review_dic[i] = review
    i += 1

with open('../Data/processed_data.json', 'w') as json_file:
    json.dump(review_dic, json_file)
