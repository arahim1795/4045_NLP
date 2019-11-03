import json
import random


def sample_business_id(dataset, threshold):
    # parameters
    data_file = "../Data/processed_data.json"

    # import data
    with open(data_file, "r") as read_file:
        reviews = json.load(read_file)

    # extract business ids
    business_ids = []
    for _, review in reviews.items():
        business_ids.append(review["business_id"])

    # sample <threshold> business ids
    business_ids_set = set(business_ids)
    sampled_ids = []
    for business_id in random.sample(business_ids_set, threshold):
        sampled_ids.append(business_id)

    # export to physical file (for reference)
    with open("id_file.csv", "w") as f:
        for business_id in sampled_ids:
            f.write(business_id + "\n")

    return sampled_ids


def extract_reviews(dataset, ids):
    # sample business ids (5)
    sampled_ids = sample_business_id(reviews, 5)

    chosen_reviews = []
    for ids in sampled_ids:
        reviews_by_id = {}
        i = 0
        for _, review in reviews.items():
            if review["business_id"] == ids:
                reviews_by_id[i] = review["text"]
                i += 1
        chosen_reviews.append(reviews_by_id)

    # export to physical file
    for i in range(len(chosen_reviews)):
        with open("chosen_business_" + str(i) + ".json", "w") as f:
            json.dump(chosen_reviews[i], f)


# import data
data_file = "processed_data.json"
with open(data_file, "r") as read_file:
    reviews = json.load(read_file)

# extract ids
sampled_ids = sample_business_id(reviews, 5)

# extract_reviews
extract_reviews(reviews, sampled_ids)
