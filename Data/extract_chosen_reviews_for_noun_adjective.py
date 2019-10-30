import json

chosen_business_ids = ["S6apFS5ghsQg69rcBvm2Qg", "mSqR24h_nKXyMhwtWSih3Q", "Jcyu0ml7rxizEA8giSH-8A", "a4GRh1TlOVhPD401mSPLZg", "7e3PZzUpG5FYOTGt3O3ePA"]

data_file = 'processed_data.json'

with open(data_file, "r") as read_file:
    reviews = json.load(read_file)

chosen_reviews = []
for _, review in reviews.items():
    if review['business_id'] in chosen_business_ids:
        chosen_reviews.append(review)

with open('chosen_reviews.json', 'w') as json_file:
    json.dump(chosen_reviews, json_file)