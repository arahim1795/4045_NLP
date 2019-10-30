import json
import random
data_file = '../Data/processed_data.json'

with open(data_file, "r") as read_file:
    reviews = json.load(read_file)

business_ids = []
for _, review in reviews.items():
    business_ids.append(review['business_id'])

business_ids_set = set(business_ids)
for business_id in random.sample(business_ids_set, 5):
    print(business_id)