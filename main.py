import nltk
import os
import stanfordnlp
import subprocess
import time

from data.f_sample_reviews import sample_business_reviews
from apps.common_functions import import_data, bundle_sentences
from apps.b_sentence_segmentor.sentence_segmentor import sentence_segmentor
from apps.c_tokenise_stem.tokenise_stem import tokenise_stem
from apps.d_pos_tagger.pos_tagger import pos_tagger
from apps.e_frequent_adjectives.frequent_adjectives import frequent_adjectives
from apps.f_noun_adjective_pair.noun_adjective_pair import summariser
from apps.g_application.negation_app import negation_app

absolute = os.path.dirname(os.path.abspath(__file__))
os.environ["CORENLP_HOME"] = (
    absolute.replace("\\", "/") + "/server/stanford-corenlp-full-2018-10-05"
)

# required libraries
# - nltk
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("stopwords")
# - stanfordnlp
stanfordnlp.download("en_ewt")
stanfordnlp.download("en_gum")

# parameters
num_business = 5

# import data
data = import_data("processed_data")  # processed_data.json
bundled_data = bundle_sentences(data)
sampled_reviews = sample_business_reviews(data, num_business)

# b: sentence_segmentation
sentence_segmentor(bundled_data)

# c: tokenise and stem
tokenise_stem(data)

# d: pos_tagging
pos_tagger(bundled_data, 5)

# e: most_frequent_adjective
frequent_adjectives(bundled_data)

# launch corenlp server
os.chdir("server/stanford-corenlp-full-2018-10-05")
server_process = subprocess.Popen(
    'java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer  -port 9000 -timeout 120000',
    shell=True,
)
time.sleep(5)
os.chdir("../..")

# f: noun_adjective_pair
summariser(num_business)

# g: application
negation_app(data)

# closer corenlp server
os.chdir("server/stanford-corenlp-full-2018-10-05")
subprocess.call(["taskkill", "/F", "/T", "/PID", str(server_process.pid)])
os.chdir("../..")
