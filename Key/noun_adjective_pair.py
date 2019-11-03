import json
from stanford_utils import get_noun_adjective_pairs_from_reviews


# import data
reviews_all = []
threshold = 5
for i in range(threshold):
    with open("../Data/chosen_business_" + str(i) + ".json", "r") as f:
        reviews_all.append(json.load(f))

# poss data thru parser
actual_noun_adjective_pairs = []
for reviews_per_business in reviews_all:
    for _, review in reviews_per_business.items():
        noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(review)
        actual_noun_adjective_pairs.extend(noun_adjective_pairs)
    print(actual_noun_adjective_pairs)

# sentences = doc.
# sents
# for sentence in sentences:
#   print(sentence)
#   print(get_ne_in_sent_from(str(sentence)))
