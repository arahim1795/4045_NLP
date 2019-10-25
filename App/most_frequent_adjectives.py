import json
import stanfordnlp
from collections import Counter

data_file = '../Data/test.json'

nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos")

sentences_list = []

with open(data_file, 'r') as json_file:
    for line in json_file:
        review = json.loads(line)
        sentences = review['text']
        sentences_list.append(sentences)

most_common_adjectives = []

for sentence in sentences_list:
    doc = nlp(sentence)
    parsed_text = {}
    for sent in doc.sentences:
        for wrd in sent.words:
            parsed_text[wrd.text] = wrd.upos
    temp_common_adjectives = []
    for key, val in parsed_text.items():
        if val == 'ADJ':
            temp_common_adjectives.append(key)
    most_common_adjectives.extend(temp_common_adjectives)

most_common_adjectives_counter = Counter(most_common_adjectives)
print(most_common_adjectives_counter.most_common(20))
