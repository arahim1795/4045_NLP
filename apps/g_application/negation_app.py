import stanfordnlp
from stanfordnlp.server import CoreNLPClient


def export_data(neg_sentences_list):
    filename = "out/g_neg_sents_results.txt"
    with open(filename, "w") as f:
        f.write("Number of negation sentences: " + str(len(neg_sentences_list)))
        for sentence in neg_sentences_list:
            f.write("\n" + str(sentence))


def negation_app(review_dic):
    # parameters
    review_list = []

    # extract reviews
    for key in review_dic:
        review = (str(review_dic.get(key, {}).get("text"))).capitalize()
        review_list.append(review)  # list of sentences regardless of reviews

    # extract sentences
    sentence_list = []
    neg_sentences_list = []
    nlp = stanfordnlp.Pipeline(processors="tokenize")

    for review in review_list:
        doc = nlp(review)
        for i in range(0, len(doc.sentences)):
            sentence_list.append(
                " ".join(["{}".format(word.text) for word in doc.sentences[i].words])
            )

    # check negation
    with CoreNLPClient(
        annotators=["tokenize", "ssplit", "pos", "depparse"],
        timeout=120000,
        memory="5G",
    ) as client:

        for sent in sentence_list:
            ann = client.annotate(sent)
            for sentence in ann.sentence:
                dependency_parse = sentence.basicDependencies
                for i in range(0, len(dependency_parse.edge)):
                    dep = dependency_parse.edge[i].dep
                    if dep == "neg":
                        neg_sentences_list.append(sent.capitalize())
                        break

    # Export
    export_data(neg_sentences_list)
