"""
Use forward, backward, and forward-backward algorithms to compute probabilities of observation sequences of Hidden Markov Model (HMM) and re-estimate parameters of HMM to maximize the probability.
"""

import re
import copy
import pickle
from math import exp
from math import log

############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    regex = re.compile('[^a-zA-Z\s]')

    with open(path) as corpus:
        string = corpus.read()
        out = regex.sub('', string)
        out = ' '.join(out.split()).lower()
    return out

def load_parameters(path):
    prob_vector = list()
    with open(path, 'rb') as handle:
        prob_li = pickle.load(handle)
        for i in xrange(len(prob_li)):
            if i == 0: # initial prob
                init_prob = prob_li[i]
                for key, val in init_prob.iteritems():
                    init_prob[key] = log(val)
                prob_vector.append(init_prob)
            else: # transition prob or emission prob
                prob = prob_li[i]
                for key, val in prob.iteritems():
                    for key2, val2 in val.iteritems():
                        prob[key][key2] = log(val2)
                prob_vector.append(prob)
    return tuple(prob_vector)

class HMM(object):

    def __init__(self, probabilities):
        self.log_prob_vector = copy.deepcopy(probabilities)
        self.prob_vector = list()

        for i in xrange(len(probabilities)):
            if i == 0: # initial
                init_prob = probabilities[i]
                for key, val in init_prob.iteritems():
                    init_prob[key] = exp(val)
                self.prob_vector.append(init_prob)
            else: # transition & emit
                prob = probabilities[i]
                for key, val in prob.iteritems():
                    for key2, val2 in val.iteritems():
                        prob[key][key2] = exp(val2)
                self.prob_vector.append(prob)

    def get_parameters(self):
        return tuple(self.prob_vector)

    def forward(self, sequence):
        seq = list(sequence)
        N = len(self.prob_vector[0])
        T = len(seq)

        init_dict = {key : val + self.log_prob_vector[2][key][seq[0]] for key, val in self.log_prob_vector[0].iteritems()}
        forward_prob = [init_dict]
        for t in xrange(1, T):

            di = dict()
            for j in xrange(1, N + 1): # j at t

                log_sum_list = []
                for i in xrange(1, N + 1): # i at t - 1
                    # alpha_t-1_i * a_ij = log(alpha_t-1_i) + log(a_ij)
                    log_sum_list.append(forward_prob[t - 1][i] + self.log_prob_vector[1][i][j]) # now in log

                alpha_tj = self.log_sum(log_sum_list)

                alpha_tj += self.log_prob_vector[2][j][seq[t]]
                di[j] = alpha_tj

            forward_prob.append(di)
        return forward_prob

    def forward_probability(self, alpha):
        log_li = alpha[-1].values()
        amax = max(log_li)
        return amax + log(sum([exp(xn - amax) for xn in log_li]))

    def backward(self, sequence):
        seq = list(sequence)
        N = len(self.prob_vector[0])
        T = len(seq)

        backward_prob = [{} for i in xrange(T)]
        backward_prob[T - 1] = {s : log(1) for s in self.log_prob_vector[0].keys()}

        for t in xrange(T - 2, -1, -1):

            di = dict()
            for i in xrange(1, N + 1):

                log_sum_list = []
                for j in xrange(1, N + 1):
                    log_sum_list.append(backward_prob[t + 1][j] + self.log_prob_vector[1][i][j] \
                                        + self.log_prob_vector[2][j][seq[t + 1]])

                beta_t_i = self.log_sum(log_sum_list)
                di[i] = beta_t_i
            backward_prob[t] = di

        return backward_prob

    def backward_probability(self, beta, sequence):
        seq = list(sequence)
        log_list = [beta + self.log_prob_vector[0][state] + self.log_prob_vector[2][state][seq[0]] for state, beta in beta[0].iteritems()]
        amax = max(log_list)
        return amax + log(sum([exp(xn - amax) for xn in log_list]))

    def xi_matrix(self, t, sequence, alpha, beta):
        N = len(self.log_prob_vector[0])
        transition = self.log_prob_vector[1]
        emission = self.log_prob_vector[2]
        seq = list(sequence)

        # calc denominator
        i_list = []
        for i in xrange(1, N + 1):
            j_list = list()
            for j in xrange(1, N + 1):
                numerator = alpha[t][i] + transition[i][j] + emission[j][seq[t + 1]] + beta[t + 1][j]
                j_list.append(numerator)
            i_list.append(j_list)

        all_list = [num for j_list in i_list for num in j_list]
        # amax = max(all_list)
        # denom_sum = amax + log(sum([exp(xn - amax) for xn in all_list]))
        denom_sum = self.log_sum(all_list)
        # calc numerator & xi matrix
        xi_matrix = dict()
        for i in xrange(1, N + 1):
            di = dict()
            for j in xrange(1, N + 1):
                xi_ij = i_list[i - 1][j - 1] - denom_sum
                di[j] = xi_ij
            xi_matrix[i] = di

        return xi_matrix

    def forward_backward(self, sequence):
        alpha = self.forward(sequence)
        beta = self.backward(sequence)
        seq = list(sequence)
        N = len(self.prob_vector[0])
        T = len(seq)
        xi_t_lookup = {t : self.xi_matrix(t, sequence, alpha, beta) for t in xrange(T - 1)}
        probabilities = [{} for i in xrange(3)]

        # reestimate initial
        xi_t0 = xi_t_lookup[0]
        gamma_t0 = dict()
        for i in xrange(1, N + 1):
            gamma_t0_i = self.log_sum([xi_t0[i][j] for j in xrange(1, N + 1)])
            gamma_t0[i] = gamma_t0_i

        probabilities[0] = gamma_t0

        # reestimate transition
        gamma_sum_a = dict()
        for t in xrange(T - 1):
            xi_t = xi_t_lookup[t]
            gamma_ti_li = list()
            di = dict()
            for i in xrange(1, N + 1):
                gamma_ti = self.log_sum([xi_t[i][j] for j in xrange(1, N + 1)])
                di[i] = gamma_ti
            gamma_sum_a[t] = di

        transition_re = dict()
        for i in xrange(1, N + 1):
            di = {}
            for j in xrange(1, N + 1):
                a_ij = self.log_sum([xi_t_lookup[t][i][j] for t in xrange(T - 1)]) - \
                        self.log_sum([gamma_sum_a[t][i] for t in xrange(T - 1)])
                di[j] = a_ij
            transition_re[i] = di

        probabilities[1] = transition_re

        # reestimate emission
        gamma_sum_b = dict()
        for t in xrange(T - 1):
            xi_t = xi_t_lookup[t]
            di = dict()
            for i in xrange(1, N + 1):
                gamma_ti = self.log_sum([xi_t[i][j] for j in xrange(1, N + 1)])
                di[i] = gamma_ti
            gamma_sum_b[t] = di
        # add when t = T - 1
        denom = self.log_sum([alpha[T - 1][j] + beta[T - 1][j] for j in xrange(1, N + 1)])
        di = dict()
        for i in xrange(1, N + 1):
            gamma_t_last = alpha[T - 1][i] + beta[T - 1][i] - denom
            di[i] = gamma_t_last
        gamma_sum_b[T - 1] = di

        all_char = self.log_prob_vector[2][1].keys()
        emission_re = dict()
        for i in xrange(1, N + 1):
            di = dict()
            denom = self.log_sum([gamma_sum_b[t][i] for t in xrange(T)])

            for k in all_char:
                numerator_li = []
                for t in xrange(T):
                    if k == seq[t]:
                        gamma_ti = gamma_sum_b[t][i]
                        numerator_li.append(gamma_ti)

                numerator = self.log_sum(numerator_li)
                di[k] = numerator - denom
            emission_re[i] = di

        probabilities[2] = emission_re
        return probabilities

    def log_sum(self, log_list):
        amax = max(log_list)
        return amax + log(sum([exp(xn - amax) for xn in log_list]))

    def update(self, sequence, cutoff_value):
        alpha = self.forward(sequence)
        incre = abs(self.forward_probability(alpha)) + cutoff_value

        while incre >= cutoff_value:
            alpha = self.forward(sequence)
            prev_prob = self.forward_probability(alpha)

            # update model
            fb = self.forward_backward(sequence)
            self.log_prob_vector = fb # CAUTION: prov_vector not update (not used)

            alpha = self.forward(sequence)
            incre = self.forward_probability(alpha) - prev_prob
