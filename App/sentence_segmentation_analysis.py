import spacy
import random
import json

data_file = '../Data/processed_data.json'

nlp = spacy.load('en_core_web_sm')

with open(data_file, 'r') as json_file:
    reviews_json_object = json.load(json_file)

reviews_list = list(reviews_json_object.values())

sampling_list = random.sample(reviews_list, 5)

results = {}

for sample in sampling_list:
    review_id = sample['review_id']
    doc = nlp(sample['text'])
    texts = [sent.text for sent in doc.sents]
    results[review_id] = texts

with open('../Data/sentence_segmentation_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4, sort_keys=True)
