import json
import random
from allennlp import pretrained
from dependency_utils import get_noun_adjective_pairs_from_allen_nlp
import spacy

data_file = '../Data/test.json'
reviews_list = []

with open(data_file, 'r') as json_file:
  for line in json_file:
    review = json.loads(line)
    reviews_list.append(review['text'])

nlp = spacy.load('en_core_web_sm')

for review in random.sample(reviews_list, 5):
  doc = nlp(review)
  sentences = doc.sents
  noun_adjective_pairs = []
  for sent in sentences:
    noun_adjective_pairs.extend(get_noun_adjective_pairs_from_allen_nlp(str(sent)))
  print(noun_adjective_pairs)
