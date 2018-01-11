"""
Build an improved spam filter with an accuracy rate of 99.5%
Features to consider include bigram, caps, digit, punc, token_len, email_header
"""


import email
import math
import os
import heapq
import string
from collections import defaultdict
from collections import Counter

############################################################
# Section 1: Probability
############################################################

# Set the following variables to True or False.
section_1_problem_1a = False
section_1_problem_1b = True
section_1_problem_1c = False

# Set the following variables to True or False.
section_1_problem_2a = True
section_1_problem_2b = False
section_1_problem_2c = False
section_1_problem_2d = False
section_1_problem_2e = True
section_1_problem_2f = False
section_1_problem_2g = False
section_1_problem_2h = True

# Set the following variables to probabilities, expressed as decimals between 0
# and 1.
section_1_problem_3a = 0.01162455
section_1_problem_3b = 0.98837545
section_1_problem_3c = 0.1087184125
section_1_problem_3d = 0.10211147958415596
section_1_problem_3e = 0.0005048688734630145
section_1_problem_3f = 7.4625e-06
section_1_problem_3g = 0.0005145525

# Show calculation steps
"""
P(A) = P(A | B, E) * P(B, E) + P(A | B, ~E) * P(B, ~E) + P(A | ~B, E) * P(~B, E) + P(A | ~B, E) *P(~B, ~E)
p_a = 0.97 * 0.01 * 0.005 + 0.95 * 0.01 * 0.995 + 0.23 * 0.99 * 0.005 + 0.001 * 0.99 * 0.995

P(~A) = P(~A|B, E)*P(B, E) + P(~A|B,~E)*P(B,~E) + P(~A|~B,E)*P(~B,E) + P(~A|~B,~E)*P(~B,~E)
p_not_a = 0.03 * 0.01 * 0.005 + 0.05 * 0.01 * 0.995 + 0.77 * 0.99 * 0.005 + 0.999 * 0.99 * 0.995

p_j = 0.85 * p_a + 0.1 * p_not_a

p_a_given_e = 0.97 * 0.01 + 0.23 * 0.99
p_e_given_a = p_a_given_e * 0.005 / p_a

p_not_a_given_b = 0.03 * 0.005 + 0.05 * 0.995
p_b_given_not_a = p_not_a_given_b * 0.01 / p_not_a

P(B,¬E,¬A,J,M) = 0.01 * 0.995 * 0.05 * 0.1 * 0.15

P(¬B,E,¬A,¬J,M) = 0.99 * 0.005 * 0.77 * 0.9 * 0.15
"""

############################################################
# Section 2: Spam Filter
############################################################
def load_tokens(email_path):
    tokens = list()

    with open(email_path) as email_file:
        email_msg = email.message_from_file(email_file)

        line_iter = email.iterators.body_line_iterator(email_msg)
        for line in line_iter:
#             """ Clean unigram tokens to contain only alphabetic & numeric"""
#             for elem in line.split():
#                 cleaned = "".join(c for c in elem if c.isalnum())
#                 if len(cleaned) > 0:
#                     tokens.append(cleaned)

            tokens.extend(line.split())
    return tokens

def load_bigrams(tokens):
    bigrams = list()
    for i in xrange(1, len(tokens)):
        bigram = " ".join([tokens[i - 1], tokens[i]])
        bigrams.append(bigram)
    return bigrams

def log_probs(email_paths, bigram, caps, digit, punc, token_len, email_header):
    """ Set smoothing factor manually """
    uni_smoothing = 1e-10 # manually set
    bi_smoothing = 1e-16 # manually set
    feature_smoothing = 1e-1 # manually set
    """ Build prob_lookup for unigrams and bigrams"""
    total_counter = Counter()

    for path in email_paths:
        tokens = load_tokens(path)
        uni_counter = Counter(tokens)
        total_counter.update(uni_counter)

        for tok in tokens:
            """ Consider capitalization feature """
            if caps:
                if tok.isupper() and len(tok) > 1:
                    total_counter['all_caps'] = total_counter.get('all_caps', 0) + 1
                elif tok.islower():
                    total_counter['all_lower'] = total_counter.get('all_lower', 0) + 1
                elif tok[0].isupper():
                    total_counter['capitalized'] = total_counter.get('capitalized', 0) + 1
                else:
                    total_counter['mixed_case'] = total_counter.get('mixed_case', 0) + 1

            """ Consider digit feature """
            if digit:
                if tok.isdigit():
                    total_counter['all_digits'] = total_counter.get('all_digits', 0) + 1
                elif any(char.isdigit() for char in tok):
                    total_counter['has_digits'] = total_counter.get('has_digits', 0) + 1
                elif tok[0].isdigit():
                    total_counter['starts_digits'] = total_counter.get('starts_digits', 0) + 1

            """ Consider punctuation feature """
            if punc:
                punc_set = set(string.punctuation)
                if all(char in punc_set for char in tok):
                    total_counter['all_punc'] = total_counter.get('all_punc', 0) + 1
                elif any(char in punc_set for char in tok):
                    total_counter['has_punc'] = total_counter.get('has_punc', 0) + 1
                elif any(char == '!' for char in tok):
                    total_counter['excl_punc'] = total_counter.get('excl_punc', 0) + 1

            """ Consider token length feature """
            if token_len:
                if len(tok) >= 15:
                    total_counter['len_long'] = total_counter.get('len_long', 0) + 1
                elif len(tok) < 5:
                    total_counter['len_short'] = total_counter.get('len_short', 0) + 1
                else:
                    total_counter['len_norm'] = total_counter.get('len_norm', 0) + 1

        """ Consider bigram feature """
        if bigram:
            bigrams = load_bigrams(tokens)
            bi_counter = Counter(bigrams)
            total_counter.update(bi_counter)

    total_distinct = len(total_counter)
    total_count = sum(total_counter.values())
    uni_unk_prob = calc_unk_prob(total_count, total_distinct, uni_smoothing)
    prob_lookup = defaultdict(lambda: uni_unk_prob)

    bi_unk_prob = calc_unk_prob(total_count, total_distinct, bi_smoothing)
    prob_lookup['unk_bigram'] = bi_unk_prob

    feature_set = {'all_caps', 'all_lower', 'capitalized', 'mixed_case', 'all_digits', 'has_digits'\
                  'starts_digits', 'all_punc', 'has_punc', 'excl_punc', 'len_long', 'len_short', 'len_norm'}

    for word in total_counter:
        word_count = total_counter[word]
        if word in feature_set: # other feature
            log_prob = calc_log_prob(word_count, total_count, total_distinct, feature_smoothing)
        elif len(word.split()) == 2: # bigram
            log_prob = calc_log_prob(word_count, total_count, total_distinct, bi_smoothing)
        else: # unigram or subject
            log_prob = calc_log_prob(word_count, total_count, total_distinct, uni_smoothing)

        prob_lookup[word] = log_prob

    return prob_lookup

def calc_log_prob(word_count, all_count, vocab_distinct, smoothing):
    return math.log((word_count + smoothing) / ((all_count) + smoothing * (vocab_distinct + 1)))

def calc_unk_prob(all_count, vocab_distinct, smoothing):
    return math.log((smoothing) / ((all_count) + smoothing * (vocab_distinct + 1)))

class SpamFilter(object):

    # Note that the initialization signature here is slightly different than the
    # one in the previous homework. In particular, any smoothing parameters used
    # by your model will have to be hard-coded in.

    def __init__(self, spam_dir, ham_dir):
        """ Control the On/Off of certain features
            Best result from: bigram, digit, token_len"""
        self.feature_bigram = True
        self.feature_caps = False
        self.feature_digit = True
        self.feature_punc = False
        self.feature_token_len = True
        self.feature_email_header = False
        """ Calculate baseline spam and ham proportions"""
        spam_paths = [spam_dir + '/' + fname for fname in os.listdir(spam_dir)]
        num_spams = len(spam_paths)
        ham_paths = [ham_dir + '/' + fname for fname in os.listdir(ham_dir)]
        num_hams = len(ham_paths)
        self.prob_is_spam = 1.0 * num_spams / (num_spams + num_hams)
        self.prob_not_spam = 1.0 * num_hams / (num_spams + num_hams)
        """ Build probability lookup """
        self.spam_prob_dict = log_probs(spam_paths, self.feature_bigram, self.feature_caps, self.feature_digit, \
                                       self.feature_punc, self.feature_token_len, self.feature_email_header)
        self.ham_prob_dict = log_probs(ham_paths, self.feature_bigram, self.feature_caps, self.feature_digit, \
                                       self.feature_punc, self.feature_token_len, self.feature_email_header)

    def is_spam(self, email_path):
        tokens = load_tokens(email_path)
        """ Consider the bigram feature """
        if self.feature_bigram:
            bigrams = load_bigrams(tokens)
            tokens.extend(bigrams)
        word_counter = Counter(tokens)

        sum_spam_log_probs = 0
        sum_ham_log_probs = 0

        for word in word_counter.keys():
            count_w = word_counter[word]
            num_gram = len(word.split())

            if num_gram == 2: # bigram
                sum_spam_log_probs += count_w * self.spam_prob_dict.get(word, self.spam_prob_dict['unk_bigram'])
                sum_ham_log_probs += count_w * self.ham_prob_dict.get(word, self.ham_prob_dict['unk_bigram'])

            elif num_gram == 1: # unigram
                sum_spam_log_probs += count_w * self.spam_prob_dict[word]
                sum_ham_log_probs += count_w * self.ham_prob_dict[word]

                """ Consider capitalization feature """
                if self.feature_caps:
                    if len(word) > 1 and word.isupper():
                        sum_spam_log_probs += count_w * self.spam_prob_dict['all_caps']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['all_caps']
                    elif len(word) > 1 and word.islower():
                        sum_spam_log_probs += count_w * self.spam_prob_dict['all_lower']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['all_lower']
                    elif len(word) > 1 and word[0].isupper():
                        sum_spam_log_probs += count_w * self.spam_prob_dict['capitalized']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['capitalized']
                    elif len(word) > 1:
                        sum_spam_log_probs += count_w * self.spam_prob_dict['mixed_case']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['mixed_case']

                """ Consider digit feature """
                if self.feature_digit:
                    if word.isdigit():
                        sum_spam_log_probs += count_w * self.spam_prob_dict['all_digits']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['all_digits']
                    elif any(char.isdigit() for char in word):
                        sum_spam_log_probs += count_w * self.spam_prob_dict['has_digits']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['has_digits']
                    elif word[0].isdigit():
                        sum_spam_log_probs += count_w * self.spam_prob_dict['starts_digits']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['starts_digits']

                """ Consider punctuation feature """
                if self.feature_punc:
                    punc_set = set(string.punctuation)
                    if all(char in punc_set for char in word):
                        sum_spam_log_probs += count_w * self.spam_prob_dict['all_punc']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['all_punc']
                    elif any(char in punc_set for char in word):
                        sum_spam_log_probs += count_w * self.spam_prob_dict['has_punc']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['has_punc']
                    elif any(char == '!' for char in word):
                        sum_spam_log_probs += count_w * self.spam_prob_dict['excl_punc']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['excl_punc']


                """ Consider token length feature """
                if self.feature_token_len:
                    if len(word) >= 15:
                        sum_spam_log_probs += count_w * self.spam_prob_dict['len_long']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['len_long']
                    elif len(word) < 5:
                        sum_spam_log_probs += count_w * self.spam_prob_dict['len_short']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['len_short']
                    else:
                        sum_spam_log_probs += count_w * self.spam_prob_dict['len_norm']
                        sum_ham_log_probs += count_w * self.ham_prob_dict['len_norm']

            elif self.feature_email_header: # num_gram > 2
                sum_spam_log_probs += count_w * self.spam_prob_dict[word]
                sum_ham_log_probs += count_w * self.ham_prob_dict[word]

        c_spam = math.log(self.prob_is_spam) + sum_spam_log_probs
        c_ham = math.log(self.prob_not_spam) + sum_ham_log_probs

        return True if c_spam > c_ham else False
