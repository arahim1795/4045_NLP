import random

seed = 10
random.seed(seed)


def sample_business_reviews(review_dic, num_business):
    # extract unique business ids
    biz_ids = set()
    biz_ids.update(value["business_id"] for key, value in review_dic.items())

    # sample business ids
    sampled_ids = []
    for biz_id in random.sample(biz_ids, num_business):
        sampled_ids.append(biz_id)

    # export sampled business ids
    with open("data/f_business_ids.txt", "w") as f:
        for biz_id in sampled_ids:
            f.write(biz_id + "\n")

    #  get reviews of sampled business ids
    reviews_sampled = []
    for biz_id in sampled_ids:
        reviews = []
        for key, value in review_dic.items():
            if value["business_id"] == biz_id:
                reviews.append(value["text"])
        reviews_sampled.append(reviews)

    # export reviews
    for i in range(len(reviews_sampled)):
        with open("data/f_business_" + str(i + 1) + ".txt", "w") as f:
            f.write(sampled_ids[i] + "\n")
            for review in reviews_sampled[i]:
                f.write(review + "\n")

    return reviews_sampled
