from App.dependency_utils import get_noun_adjective_pairs_from_allen_nlp

def test_basic_noun_adjective_pair_is_returned():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("John is cool.")
    assert ("John", "cool") in noun_adjective_pairs

def test_correct_noun_adjective_pair_is_returned_for_sentence_with_connected_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("John is cool and calm.")
    assert("John", "cool") in noun_adjective_pairs
    assert("John", "calm") in  noun_adjective_pairs

def test_correct_noun_adjective_pair_is_returned_for_sentence_with_long_name():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("Tan Ah Kao is cool and calm.")
    assert("Tan Ah Kao", "cool") in noun_adjective_pairs

def test_correct_noun_adjective_pair_is_returned_for_sentence_sentences_with_adverbs_attached_to_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("Tan Ah Kao is very cool and very calm.")
    assert("Tan Ah Kao", "very cool") in noun_adjective_pairs 
    assert("Tan Ah Kao", "very calm") in noun_adjective_pairs
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("Tan Ah Kao is very cool but not very calm.")
    assert("Tan Ah Kao", "very cool") in noun_adjective_pairs 
    assert("Tan Ah Kao", "not very calm") in noun_adjective_pairs

def test_correct_noun_adjective_pair_is_returned_for_sentence_sentences_with_multiple_nouns_and_one_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("The food, music and wine were great.")
    assert("food", "great") in noun_adjective_pairs 
    assert("music", "great") in noun_adjective_pairs 
    assert("wine", "great") in noun_adjective_pairs
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("Lebron James, Yao Ming and Kobe Bryant are very tall.")
    assert("Lebron James", "very tall") in noun_adjective_pairs 
    assert("Yao Ming", "very tall") in noun_adjective_pairs
    assert("Kobe Bryant", "very tall") in noun_adjective_pairs

def test_correct_noun_adjective_pair_is_returned_for_sentence_sentences_with_multiple_nouns_and_multiple_adjectives():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("The food, music and wine were great, delicious but not very cheap.")
    assert("food", "great") in noun_adjective_pairs 
    assert("music", "great") in noun_adjective_pairs 
    assert("wine", "great") in noun_adjective_pairs
    assert("food", "delicious") in noun_adjective_pairs 
    assert("music", "delicious") in noun_adjective_pairs 
    assert("wine", "delicious") in noun_adjective_pairs
    assert("food", "great") in noun_adjective_pairs 
    assert("music", "great") in noun_adjective_pairs 
    assert("wine", "great") in noun_adjective_pairs
    assert("food", "not very cheap") in noun_adjective_pairs 
    assert("music", "not very cheap") in noun_adjective_pairs 
    assert("wine", "not very cheap") in noun_adjective_pairs
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("Lebron James and Kobe Bryant are not very short and not very ugly.")
    assert("Lebron James", "not very short") in noun_adjective_pairs 
    assert("Kobe Bryant", "not very short") in noun_adjective_pairs
    assert("Lebron James", "not very ugly") in noun_adjective_pairs 
    assert("Kobe Bryant", "not very ugly") in noun_adjective_pairs

def test_correct_noun_adjective_pair_is_returned_for_seperate_noun_and_adjective_pair():
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("The food is great and wine is delicious.")
    assert("food", "great") in noun_adjective_pairs
    assert("wine", "delicious") in noun_adjective_pairs
    assert len(noun_adjective_pairs) == 2
    noun_adjective_pairs = get_noun_adjective_pairs_from_allen_nlp("Nice atmosphere and very great music.")
    assert("atmosphere", "Nice") in noun_adjective_pairs
    assert("music", "very great") in noun_adjective_pairs







 