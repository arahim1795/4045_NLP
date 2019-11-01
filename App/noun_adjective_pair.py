import json
import random
import spacy
import time
from named_entity_recognizer import get_ne_in_sent_from
from stanford_dependency_utils import get_noun_adjective_pairs
from stanford_dependency_utils_v2 import get_noun_adjective_pairs_from_reviews

# stanfordnlp.download("en_gum")

# nlp = spacy.load('en_core_web_sm')

data_file = '../Data/chosen_reviews.json'
reviews_list = []

with open(data_file, 'r') as json_file:
    reviews_list = json.load(json_file)
    # reviews_list.append(review['text'])
start = time.time()
actual_noun_adjective_pairs = []
for review in reviews_list:
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(review['text'])
    actual_noun_adjective_pairs.extend(noun_adjective_pairs)
  # sentences = doc.
  # sents
  # for sentence in sentences:
  #   print(sentence)
  #   print(get_ne_in_sent_from(str(sentence)))
print(actual_noun_adjective_pairs)
end = time.time()
print(end - start)



