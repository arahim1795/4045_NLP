import json
import spacy
import random

data_file = '../Data/test.json'
nlp = spacy.load('en_core_web_sm')

sentences_list = []

with open(data_file, 'r') as json_file:
    for line in json_file:
        review = json.loads(line)
        sentences = review['text']
        sentences_list.append(sentences)

random_5_sentences = random.sample(sentences_list, 5)
for sentence in random_5_sentences:
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.tag_)