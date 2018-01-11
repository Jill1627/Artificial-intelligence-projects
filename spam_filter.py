"""
Implement a basic spam filter using Naive Bayes Classification
"""


import email
import math
import os
import heapq
from collections import defaultdict
from collections import Counter

############################################################
# Section 1: Spam Filter
############################################################

def load_tokens(email_path):
    tokens = list()
    with open(email_path) as email_file:
        email_msg = email.message_from_file(email_file)
        line_iter = email.iterators.body_line_iterator(email_msg)
        for line in line_iter:
            tokens.extend(line.split())
    return tokens

def log_probs(email_paths, smoothing):
    total_counter = Counter()

    for path in email_paths:
        tokens = load_tokens(path)
        word_counter = Counter(tokens)
        total_counter.update(word_counter)

    total_distinct = len(total_counter)
    total_count = sum(total_counter.values())

    unk_prob = calc_unk_prob(total_count, total_distinct, smoothing)
    prob_lookup = defaultdict(lambda: unk_prob)

    for word in total_counter:
        word_count = total_counter[word]
        log_prob = calc_log_prob(word_count, total_count, total_distinct, smoothing)
        prob_lookup[word] = log_prob

    return prob_lookup

def calc_log_prob(word_count, all_count, vocab_distinct, smoothing):
    return math.log((word_count + smoothing) / ((all_count) + smoothing * (vocab_distinct + 1)))

def calc_unk_prob(all_count, vocab_distinct, smoothing):
    return math.log((smoothing) / ((all_count) + smoothing * (vocab_distinct + 1)))

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):

        spam_paths = [spam_dir + '/' + fname for fname in os.listdir(spam_dir)]
        num_spams = len(spam_paths)

        ham_paths = [ham_dir + '/' + fname for fname in os.listdir(ham_dir)]
        num_hams = len(ham_paths)

        self.spam_prob_dict = log_probs(spam_paths, smoothing)
        self.ham_prob_dict = log_probs(ham_paths, smoothing)
        self.prob_is_spam = 1.0 * num_spams / (num_spams + num_hams)
        self.prob_not_spam = 1.0 * num_hams / (num_spams + num_hams)

    def is_spam(self, email_path):
        tokens = load_tokens(email_path)
        word_counter = Counter(tokens)

        sum_spam_log_probs = 0
        sum_ham_log_probs = 0

        for word in word_counter.keys():
            count_w = word_counter[word]
            sum_spam_log_probs += count_w * self.spam_prob_dict[word]
            sum_ham_log_probs += count_w * self.ham_prob_dict[word]

        c_spam = math.log(self.prob_is_spam) + sum_spam_log_probs
        c_ham = math.log(self.prob_not_spam) + sum_ham_log_probs
        return True if c_spam > c_ham else False

    def most_indicative_spam(self, n):
        mutual_words = set(self.spam_prob_dict.keys()).intersection(set(self.ham_prob_dict.keys()))

        word_indication = list()
        for word in mutual_words:
            p_w = math.exp(self.spam_prob_dict[word]) * self.prob_is_spam \
            + math.exp(self.ham_prob_dict[word]) * self.prob_not_spam

            indication = self.spam_prob_dict[word] - math.log(p_w)

            word_indication.append((word, indication))

        nlarge = heapq.nlargest(n, word_indication, key = lambda i : i[1])
        return [tup[0] for tup in nlarge]

    def most_indicative_ham(self, n):
        mutual_words = set(self.spam_prob_dict.keys()).intersection(set(self.ham_prob_dict.keys()))

        word_indication = list()
        for word in mutual_words:
            p_w = math.exp(self.spam_prob_dict[word]) * self.prob_is_spam \
            + math.exp(self.ham_prob_dict[word]) * self.prob_not_spam

            indication = self.ham_prob_dict[word] - math.log(p_w)

            word_indication.append((word, indication))

        nlarge = heapq.nlargest(n, word_indication, key = lambda i : i[1])
        return [tup[0] for tup in nlarge]
