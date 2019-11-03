import json
import random
import spacy
import time
from stanford_dependency_utils_v2 import get_noun_adjective_pairs_from_reviews
from collections import Counter
from stanford_core_nlp_dependency_utils import get_noun_adjective_pairs_from_reviews_from_core_nlp


# stanfordnlp.download("en_gum")

# nlp = spacy.load('en_core_web_sm')

data_file = '../Data/chosen_reviews.json'
reviews_list = []

with open(data_file, 'r') as json_file:
    reviews_list = json.load(json_file)

start = time.time()
result = get_noun_adjective_pairs_from_reviews_from_core_nlp(reviews_list)
for business_id, noun_adjective_pairs in result.items():
    noun_adjective_pairs_counter = Counter(noun_adjective_pairs)
    print(business_id)
    print(noun_adjective_pairs_counter.most_common(5))
end = time.time()
print(end - start)

start = time.time()
actual_noun_adjective_pairs = []
reviews = [review['text'] for review in reviews_list]
business_ids = [review['business_id'] for review in reviews_list]
count = 0
review_dict= {}
for review in reviews:
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(review)
    if business_ids[count] in review_dict:
        actual_noun_adjective_pairs = review_dict[business_ids[count]]
        actual_noun_adjective_pairs.extend(noun_adjective_pairs)
    else:
        review_dict[business_ids[count]] = noun_adjective_pairs
    count += 1

for business_id, noun_adjective_pairs in review_dict.items():
    noun_adjective_pairs_counter = Counter(noun_adjective_pairs)
    print(business_id)
    print(noun_adjective_pairs_counter.most_common(5))

end = time.time()
print(end - start)



