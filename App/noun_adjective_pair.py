import json
import random
import spacy
import time
from named_entity_recognizer import get_ne_in_sent_from
from stanford_dependency_utils import get_noun_adjective_pairs
from stanford_dependency_utils_v2 import get_noun_adjective_pairs_from_reviews

# stanfordnlp.download("en_gum")

# nlp = spacy.load('en_core_web_sm')

data_file = '../Data/test.json'
reviews_list = []

with open(data_file, 'r') as json_file:
    for line in json_file:
        review = json.loads(line)
        reviews_list.append(review['text'])

for review in reviews_list:
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(review)
    enter = input("Press a key to continue: ")

  # sentences = doc.sents
  # for sentence in sentences:
  #   print(sentence)
  #   print(get_ne_in_sent_from(str(sentence)))


