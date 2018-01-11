"""
Build a language model to generate random text resembling a source document
"""

import string
import random
from collections import Counter
from math import log
from math import exp

############################################################
# Section 1: Markov Models
############################################################

def tokenize(text):
    res = list()
    tokens = text.split()
    punc_set = set(string.punctuation)
    for tok in tokens:
        if tok.isalnum():
            res.append(tok)
        else:
            temp = list()
            for c in tok:
                if c not in punc_set:
                    temp.append(c)
                else:
                    if len(temp) > 0:
                        res.append(''.join(temp))
                    res.append(c)
                    temp = list()
            if len(temp) > 0:
                res.append(''.join(temp))
    return res


def ngrams(n, tokens):
    res = list()
    for i in xrange(len(tokens)):
        context = list()
        for j in xrange(i - n + 1, i):
            if j < 0:
                context.append('<START>')
            else:
                context.append(tokens[j])

        tup = (tuple(context), tokens[i])
        res.append(tup)

    context = list()
    size = len(tokens)
    for j in xrange(size - n + 1, size):
            if j < 0:
                context.append('<START>')
            else:
                context.append(tokens[j])
    res.append((tuple(context), '<END>'))
    return res

class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.ngram_counter = Counter()
        self.context_counter = Counter()
        self.ngram_map = dict()

    def update(self, sentence):
        tokens = tokenize(sentence)
        n_grams = ngrams(self.n, tokens)

        self.ngram_counter.update(Counter(n_grams))

        context = list()
        for tup in n_grams:
            context.append(tup[0])
            if tup[0] in self.ngram_map:
                self.ngram_map[tup[0]].add(tup[1])
            else:
                self.ngram_map[tup[0]] = {tup[1]} # use set or list?
        self.context_counter.update(Counter(context))

    def prob(self, context, token):
        ngram_count = self.ngram_counter[(context, token)]
        context_count = self.context_counter[context]
        return 1.0 * ngram_count / context_count

    def random_token(self, context):
        tokens = self.ngram_map[context]
        tokens = list(tokens)
        tokens.sort() # sort tokens lexico

        cdf = 0
        r = random.random()
        for i in xrange(len(tokens)):
            cdf += self.prob(context, tokens[i])
            if cdf > r:
                return tokens[i]
        return tokens[-1]

    def random_text(self, token_count):
        tok_list = []
        start = tuple('<START>' for i in xrange(self.n - 1))
        context = start

        for i in xrange(token_count):
            tok = self.random_token(context)
            tok_list.append(tok)
            if tok == '<END>':
                context = start
            elif self.n - 1 > 0:
                contx_list = list(context)
                contx_list = contx_list[1 : ]
                contx_list.append(tok)
                context = tuple(contx_list)
        return ' '.join(tok_list)

    def perplexity(self, sentence):
        sent_list = tokenize(sentence)
        m = len(sent_list)

        n_grams = ngrams(self.n, sent_list)
        log_prob = 0

        for e in n_grams:
            log_prob += log(self.prob(e[0], e[1]))

        return 1.0 / (pow(exp(log_prob), 1.0 / (m + 1)))

def create_ngram_model(n, path):
    ngram_model = NgramModel(n)
    with open(path) as doc:
        for line in doc:
            ngram_model.update(line)
    return ngram_model
