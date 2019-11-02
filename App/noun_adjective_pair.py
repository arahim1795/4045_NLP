import json
import random
import spacy
import time
from stanford_dependency_utils_v2 import get_noun_adjective_pairs_from_reviews
from stanford_core_nlp_dependency_utils import get_noun_adjective_pairs_from_reviews


# stanfordnlp.download("en_gum")

# nlp = spacy.load('en_core_web_sm')

data_file = '../Data/chosen_reviews.json'
reviews_list = []

with open(data_file, 'r') as json_file:
    reviews_list = json.load(json_file)
    # reviews_list.append(review['text'])
start = time.time()
actual_noun_adjective_pairs = []
reviews = [review['text'] for review in reviews_list]
get_noun_adjective_pairs_from_reviews(reviews)
end = time.time()
print(end - start)



