import json
import spacy
import random
import stanfordnlp

data_file = '../Data/test.json'
# nlp = spacy.load('en_core_web_sm')

nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos")

sentences_list = []

with open(data_file, 'r') as json_file:
    for line in json_file:
        review = json.loads(line)
        sentences = review['text']
        sentences_list.append(sentences)

random_5_sentences = random.sample(sentences_list, 5)

doc = nlp("#MAGA")
parsed_text = {}
for sent in doc.sentences:
    for wrd in sent.words:
        parsed_text[wrd.text] = wrd.upos

print(parsed_text)


# for sentence in random_5_sentences:
#     doc = nlp(sentence)
#     parsed_text = {}
#     for sent in doc.sentences:
#         for wrd in sent.words:
#             parsed_text[wrd.text] = wrd.upos
#     print(parsed_text)
    # for token in doc:
    #     print(token.text, token.tag_)
