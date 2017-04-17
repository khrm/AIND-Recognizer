import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # The recognizer
    # return probabilities, guesses
    keys = test_set.get_all_sequences().keys()
    for key in keys:
        X, lengths = test_set.get_item_Xlengths(key)
        prob = {}
        guess = None
        best_score = float('-inf')
        for word, model in models.items():
            try:
                cur_score = model.score(X, lengths)
                prob[word] = cur_score
            except:
                prob[word] = None
                cur_score = float('-inf')
            if cur_score > best_score:
                best_score = cur_score
                guess = word
        probabilities.append(prob)
        guesses.append(guess)
    return probabilities, guesses
