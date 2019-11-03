import json
import spacy
import random
import stanfordnlp

data_file = '../Data/test.json'
# nlp = spacy.load('en_core_web_sm')

nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos")

reviews_list = []

with open(data_file, 'r') as json_file:
    for line in json_file:
        review = json.loads(line)
        sentences = review['text']
        reviews_list.append(sentences)

sentences_list = []

for review in reviews_list:
    doc =nlp(review)
    sentences_list.extend([sent for sent in doc.sentences])


random_5_sentences = random.sample(sentences_list, 5)

parsed_texts = []
for sentence in random_5_sentences:
    parsed_text = {}
    for i in range(len(sentence.words)):
        parsed_text[i] = (sentence.words[i].text, sentence.words[i].xpos)
    parsed_texts.append(parsed_text)

with open('../Data/parsed_pos_tags.json', 'w') as json_file:
    json.dump(parsed_texts, json_file, indent=4, sort_keys=True)
