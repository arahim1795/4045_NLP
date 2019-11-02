import sys
sys.path.append('..')
from App.stanford_core_nlp_dependency_utils import get_noun_adjective_pairs_from_reviews

def test_basic_noun_adjective_pair_is_returned():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["John is cool."])
    assert ("John", "cool") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 1

def test_correct_noun_adjective_pair_is_returned_for_sentence_with_connected_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["John is cool and calm."])
    assert("John", "cool") in noun_adjective_pairs
    assert("John", "calm") in  noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2

def test_correct_noun_adjective_pair_is_returned_for_sentence_with_long_name():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["Kim Jong Un is cool and calm."])
    assert("Kim Jong Un", "cool") in noun_adjective_pairs
    assert("Kim Jong Un", "calm") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2

def test_correct_noun_adjective_pair_is_returned_for_sentence_sentences_with_adverbs_attached_to_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["Kim Jong Un is very cool and very calm."])
    assert("Kim Jong Un", "very cool") in noun_adjective_pairs 
    assert("Kim Jong Un", "very calm") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["Kim Jong Un is very cool but not very calm."])
    assert("Kim Jong Un", "very cool") in noun_adjective_pairs 
    assert("Kim Jong Un", "not very calm") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2

def test_correct_noun_adjective_pair_is_returned_for_sentence_sentences_with_multiple_nouns_and_one_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["The food, music and wine were great."])
    assert("food", "great") in noun_adjective_pairs 
    assert("music", "great") in noun_adjective_pairs 
    assert("wine", "great") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 3
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["Lebron James, Yao Ming and Kobe Bryant are very tall."])
    assert("Lebron James", "very tall") in noun_adjective_pairs 
    assert("Yao Ming", "very tall") in noun_adjective_pairs
    assert("Kobe Bryant", "very tall") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 3

def test_correct_noun_adjective_pair_is_returned_for_sentence_sentences_with_multiple_nouns_and_multiple_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["The food, music and wine were great, delicious but not very cheap."])
    assert("food", "great") in noun_adjective_pairs 
    assert("music", "great") in noun_adjective_pairs 
    assert("wine", "great") in noun_adjective_pairs
    assert("food", "delicious") in noun_adjective_pairs 
    assert("music", "delicious") in noun_adjective_pairs 
    assert("wine", "delicious") in noun_adjective_pairs
    assert("food", "not very cheap") in noun_adjective_pairs 
    assert("music", "not very cheap") in noun_adjective_pairs 
    assert("wine", "not very cheap") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 9
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["Lebron James and Kobe Bryant are not very short and not very ugly."])
    assert("Lebron James", "not very short") in noun_adjective_pairs 
    assert("Kobe Bryant", "not very short") in noun_adjective_pairs
    assert("Lebron James", "not very ugly") in noun_adjective_pairs 
    assert("Kobe Bryant", "not very ugly") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 4

def test_correct_noun_adjective_pair_is_returned_for_seperate_noun_and_adjective_pair():
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["The food is great and wine is delicious."])
    assert("food", "great") in noun_adjective_pairs
    assert("wine", "delicious") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2
    noun_adjective_pairs = get_noun_adjective_pairs_from_reviews(["Nice atmosphere and very great music."])
    assert("atmosphere", "nice") in noun_adjective_pairs
    assert("music", "very great") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2