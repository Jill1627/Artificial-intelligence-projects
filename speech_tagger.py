############################################################
# CIS 521: Homework 8
############################################################

student_name = "He Gao"

############################################################
# Imports
############################################################

from collections import Counter
from math import log
import operator

############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    res = list()
    with open(path) as corpus:
        for line in corpus:
            tokens = line.split()
            tup_list = list()

            for tok in tokens:
                tok_list = tok.split('=')
                tup_list.append((tok_list[0], tok_list[1]))
            res.append(tup_list)
    return res


def load_unk_probs(probs, smoothing, tags_counter, tag_words_mapping):
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X']
    for pos in pos_list:
        all_count = tags_counter[pos]
        distinct_count = len(list(tag_words_mapping[pos]))
        unk_prob = 1.0 * smoothing / ((all_count) + smoothing * (distinct_count + 1))
        tup = ('<UNK>', pos)
        probs[tup] = unk_prob

class Tagger(object):

    def __init__(self, sentences):

        self.all_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X']
        smoothing = 1e-5

        num_sentences = len(sentences)

        # calc initial probs, with smoothing
        start_tags = [sen[0][1] for sen in sentences]
        start_tags_counter = Counter(start_tags)
        self.initial_probs = {start : 1.0 * start_tags_counter[start] / num_sentences for start in start_tags_counter}

        if len(start_tags_counter) < 12:
            self.initial_probs['<UNK>'] = 1.0 * smoothing / ((sum(start_tags_counter.values())) + smoothing * (len(start_tags_counter) + 1))

        # calc transition probs, with No smoothing
        tag_unigrams = [token[1] for sen in sentences for token in sen]
        tag_bigrams = [(sen[i - 1][1], sen[i][1]) for sen in sentences for i in xrange(1, len(sen))]
        tag_unigrams_counter = Counter(tag_unigrams)
        tag_bigrams_counter = Counter(tag_bigrams)
        self.transition_probs = {bigram : 1.0 * tag_bigrams_counter[bigram] / tag_unigrams_counter[bigram[0]] for bigram in tag_bigrams_counter}

        # calc emission probs, with smoothing
        word_tag_counter = Counter([tok for sen in sentences for tok in sen])
        self.emission_probs = {word_tag : 1.0 * word_tag_counter[word_tag] / tag_unigrams_counter[word_tag[1]] for word_tag in word_tag_counter}

        self.word_tags_mapping = dict()
        self.tag_words_mapping = dict()
        for sen in sentences:
            for tok in sen:
                word = tok[0]
                tag = tok[1]
                if word in self.word_tags_mapping:
                    self.word_tags_mapping[word].add(tag)
                else:
                    self.word_tags_mapping[word] = {tag}
                if tag in self.tag_words_mapping:
                    self.tag_words_mapping[tag].add(word)
                else:
                    self.tag_words_mapping[tag] = {word}

        # load unk probs
        load_unk_probs(self.emission_probs, smoothing, tag_unigrams_counter, self.tag_words_mapping)

    def most_probable_tags(self, tokens):
        res = list()
        for tok in tokens:
            max_tag = self.argmax_tag(tok)
            res.append(max_tag)
        return res

    def argmax_tag(self, tok):
        emit = 0
        most_prob_tag = ''
        if tok not in self.word_tags_mapping:
            for tag in self.all_tags:
                p = self.emission_probs[('<UNK>', tag)]
                if p > emit:
                    emit = p
                    most_prob_tag = tag
            return most_prob_tag

        for tag in self.word_tags_mapping[tok]:
            if self.emission_probs[(tok, tag)] > emit:
                emit = self.emission_probs[(tok, tag)]
                most_prob_tag = tag
        return most_prob_tag

    def viterbi_tags(self, tokens):
        backpointers = list()
        initials = self.get_initial(tokens[0])
        prev_probs = {k: log(v) for k, v in initials.items()}

        for i in xrange(1, len(tokens)):

            tok = tokens[i]
            backpointer_dict = dict()
            prev_probs_copy = dict()

            for tag in self.all_tags:

                max_backpointer_prob = -float('inf')
                backpointer = ''

                for predecessor in self.all_tags:

#                     backpointer_prob = prev_probs[predecessor] * self.transition_probs[(predecessor, tag)]
                    backpointer_prob = prev_probs[predecessor] + log(self.transition_probs[(predecessor, tag)]) # log space
                    if backpointer_prob > max_backpointer_prob:
                        max_backpointer_prob = backpointer_prob
                        backpointer = predecessor
                backpointer_dict[tag] = backpointer

                tup = (tok, tag) if (tok, tag) in self.emission_probs else ('<UNK>', tag)
#                 prev_probs_copy[tag] = max_backpointer_prob * self.emission_probs[tup]
                prev_probs_copy[tag] = max_backpointer_prob + log(self.emission_probs[tup]) # log space

            prev_probs = dict(prev_probs_copy)
            backpointers.append(backpointer_dict)

        # now find out the argmax in prev_probs
        final_tag = max(prev_probs.iteritems(), key = operator.itemgetter(1))[0]

        tags_reversed = [final_tag]
        for i in xrange(len(backpointers) - 1, -1, -1):
            pred = backpointers[i][final_tag]
            tags_reversed.append(pred)
            final_tag = pred
        return list(reversed(tags_reversed))

    def get_initial(self, start_tok): # no log
        initial_probs = dict()
        for tag in self.all_tags:
            tup = (start_tok, tag) if (start_tok, tag) in self.emission_probs else ('<UNK>', tag)
            init = self.initial_probs[tag] if tag in self.initial_probs else self.initial_probs['<UNK>']
            starting = init * self.emission_probs[tup]
            initial_probs[tag] = starting
        return initial_probs

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
10 hours
"""

feedback_question_2 = """
building trellis in viterbi algorithm and adding in smoothing, just implementation stuff
"""

feedback_question_3 = """
A bit lack of guidance in this assignment, need to spend time to design and later refactor
"""
