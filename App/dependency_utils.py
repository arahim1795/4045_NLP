from allennlp.predictors.predictor import Predictor
import networkx as nx

INTERESTED_NOUN_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$']
INTERESTED_ADJ_POS = ['JJ', 'JJR', 'JJS']
INTERESTED_ADVERB_POS = ['RB', 'RBR', 'RBS']
CORRECT_ADVERB_DEPENDENCIES = ['neg', 'advmod']
CONJUNCTION = ['conj']
ROOT_NODE_INDEX = -1

def get_dependency_tree_from_sentence(sentence):
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    dependency_tree = predictor.predict(
        sentence=sentence
    )
    return dependency_tree

def get_noun_adjective_pairs_from_allen_nlp(sentence):
    print(sentence)
    dependency_tree_dict = get_dependency_tree_from_sentence(sentence)
    G = create_graph_from_dependency_tree_dict(dependency_tree_dict)
    noun_adjective_pairs = get_noun_adjective_pairs_from_graph(G, dependency_tree_dict)
    return noun_adjective_pairs

def add_edge_from_first_node_to_second_node(G, first_node, second_node):
    if not G.has_node(first_node):
        G.add_node(first_node)
    if not G.has_node(second_node):
        G.add_node(second_node)
    G.add_edge(first_node, second_node)
    
def create_graph_from_dependency_tree_dict(dependency_tree_dict):
    """
    first level after the root node will contain nouns.
    second level after the root node will contain adjectives.
    third level after the root node will contain adverbs.
    """
    print(dependency_tree_dict)
    predicted_heads = dependency_tree_dict['predicted_heads']
    predicted_dependencies = dependency_tree_dict['predicted_dependencies']
    pos = dependency_tree_dict['pos']
    G = nx.DiGraph()
    G.add_node(ROOT_NODE_INDEX)
    for i in range(len(dependency_tree_dict['words'])):
        # predicted_heads always return 0 for root and return actual index + 1 for the rest of headers
        if predicted_heads[i] == 0:
            continue
        if pos[i] in INTERESTED_NOUN_POS:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_ADJ_POS:
                add_edge_from_first_node_to_second_node(G, i, head_index)
                add_edge_from_first_node_to_second_node(G, ROOT_NODE_INDEX, i)
            elif pos[head_index] in INTERESTED_NOUN_POS and predicted_dependencies[i] in CONJUNCTION:
                add_edge_from_first_node_to_second_node(G, head_index, i)
        if pos[i] in INTERESTED_ADJ_POS:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_NOUN_POS:
                add_edge_from_first_node_to_second_node(G, head_index, i)
                add_edge_from_first_node_to_second_node(G, ROOT_NODE_INDEX, head_index)
            # elif pos[head_index] in INTERESTED_ADJ_POS:

                # noun_index = predicted_heads[head_index] - 1
                # add_edge_from_first_node_to_second_node(G, ROOT_NODE_INDEX, noun_index)
                # add_edge_from_first_node_to_second_node(G, noun_index, head_index)
                # add_edge_from_first_node_to_second_node(G, noun_index, i)

                # noun_index = list(G.predecessors(head_index))[0]
                # add_edge_from_first_node_to_second_node(G, noun_index, i)
                # add_edge_from_first_node_to_second_node(G, ROOT_NODE_INDEX, noun_index)
        if pos[i] in INTERESTED_ADVERB_POS and predicted_dependencies[i] in CORRECT_ADVERB_DEPENDENCIES:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_ADJ_POS:
                add_edge_from_first_node_to_second_node(G, head_index, i)
    for i in range(len(dependency_tree_dict['words'])):
        if pos[i] in INTERESTED_ADJ_POS:
            head_index = predicted_heads[i] - 1
            if pos[head_index] in INTERESTED_NOUN_POS:
                continue
            if G.has_node(i) and len(list(G.predecessors(i))) > 0 and pos[list(G.predecessors(i))[0]] in INTERESTED_NOUN_POS:
                continue
            while not (G.has_node(head_index) and len(list(G.predecessors(head_index))) > 0 and pos[list(G.predecessors(head_index))[0]] in INTERESTED_NOUN_POS):
                head_index = predicted_heads[head_index] - 1
                if head_index == -1:
                    break
            if head_index != -1:
                head_index = list(G.predecessors(head_index))[0]
                add_edge_from_first_node_to_second_node(G, head_index, i)
    return G

def get_noun_groupings(dependency_tree_dict, noun_index):
    words = dependency_tree_dict['words']
    pos = dependency_tree_dict['pos']
    nouns = []
    for i in range(noun_index, -1, -1):
        if pos[i] not in INTERESTED_NOUN_POS:
            break
        nouns.append(words[i])
    nouns.reverse()
    if len(nouns) == 1:
        noun_groupings = nouns[0]
    else:
        noun_groupings = ' '.join(nouns)
    return noun_groupings


def get_noun_adjective_pairs_from_graph(G, dependency_tree_dict):
    """
    (noun, adjective) tuple will be returned
    """
    words = dependency_tree_dict['words']
    pos = dependency_tree_dict['pos']
    noun_adjective_pairs = []
    noun_indexes =  G.successors(ROOT_NODE_INDEX)
    for noun_index in noun_indexes:
        nouns = []
        # Consecutive noun words are counted as one big noun word
        # The last index will be stored in the graph so iterate backwards
        noun_groupings = get_noun_groupings(dependency_tree_dict, noun_index)
        indexes = list(G.successors(noun_index))
        adjective_indexes = [i for i in indexes if pos[i] in INTERESTED_ADJ_POS]
        # adjective_indexes = G.successors(noun_index)
        temp_noun_indexes = [i for i in indexes if pos[i] in INTERESTED_NOUN_POS]
        for adjective_index in adjective_indexes:
            adverb_indexes = list(G.successors(adjective_index))
            if len(adverb_indexes) == 0:
                noun_adjective_pairs.append((noun_groupings, words[adjective_index]))
                for temp_noun_index in temp_noun_indexes:
                    temp_noun_groupings = get_noun_groupings(dependency_tree_dict, temp_noun_index)
                    noun_adjective_pairs.append((temp_noun_groupings, words[adjective_index]))
            else:
                adverbs = []
                adverb_indexes.sort()
                if adverb_indexes[-1] < adjective_index:
                    for i in range(adverb_indexes[0] - 1, -1, -1):
                        #Used to handle cases in sentences like 'Tan Ah Kao is very cool but not very calm'
                        #where not is attached to cool instead of calm
                        if pos[i] not in INTERESTED_ADVERB_POS:
                            break
                        adverb_indexes.insert(0, i)
                    for adverb_index in adverb_indexes:
                        adverbs.append(words[adverb_index])
                    adjective_groupings = ' '.join(adverbs)
                    adjective_groupings += " " + words[adjective_index]
                else:
                    adjective_groupings = words[adjective_index]
                noun_adjective_pairs.append((noun_groupings, adjective_groupings))
                for temp_noun_index in temp_noun_indexes:
                    temp_noun_groupings = get_noun_groupings(dependency_tree_dict, temp_noun_index)
                    noun_adjective_pairs.append((temp_noun_groupings, adjective_groupings))
    return noun_adjective_pairs



