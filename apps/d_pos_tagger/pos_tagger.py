import json
import random
import stanfordnlp


def pos_tagger(bundled_data, threshold):

    # combine list of sentences
    reviews = []
    for review in bundled_data:
        reviews.extend(review)

    # parser
    parser = stanfordnlp.Pipeline(processors="tokenize,mwt,lemma,pos")

    # extract sentences from reviews
    sentences_list = []
    for review in reviews:
        doc = parser(review)
        sentences_list.extend([sentence for sentence in doc.sentences])

    sampled_sentences = random.sample(sentences_list, threshold)

    parsed_texts = []
    for sentence in sampled_sentences:
        parsed_text = {}
        for i in range(len(sentence.words)):
            parsed_text[i] = (sentence.words[i].text, sentence.words[i].xpos)
        parsed_texts.append(parsed_text)

    with open("out/d_pos_tagged.json", "w") as json_file:
        json.dump(parsed_texts, json_file, indent=4, sort_keys=True)
