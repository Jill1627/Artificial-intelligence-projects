"""
Implement two varieties of standard perceptrons: binary perceptron and MulticlassPerceptron.
"""

import homework9_data as data

############################################################
# Section 1: Perceptrons
############################################################

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):

        num_feat = max({int(k[1:]) for exp in examples for k in exp[0].keys()})
        self.w = {'x' + str(i + 1) : 0 for i in xrange(num_feat)}

        for i in xrange(iterations):
            for exp in examples:
                x_feat = exp[0]

                y = exp[1]
                y_pred = sum([x_feat[x] * self.w[x] for x in x_feat]) # skipping absent feature
                if (y_pred == 0) or (y_pred > 0) != y:
                    for xi in x_feat:
                        self.w[xi] = self.w[xi] + x_feat[xi] if y > 0 else self.w[xi] - x_feat[xi]

    def predict(self, x):
        y_pred = sum([x[feat] * self.w[feat] for feat in x])
        return y_pred > 0

class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        # initialize weight vectors
        labels = set([exp[1] for exp in examples])
        num_feat = max({int(k[1:]) for exp in examples for k in exp[0].keys()})
        vector = {'x' + str(i + 1) : 0 for i in xrange(num_feat)}

        self.w_vectors = {label : dict(vector) for label in labels}
        # loop
        for i in xrange(iterations):
            for exp in examples:
                (label, y_pred) = self.argmax(exp[0])

                if y_pred != exp[1]:
                    for feat in exp[0]:
                        self.w_vectors[exp[1]][feat] += exp[0][feat]
                        self.w_vectors[label][feat] -= exp[0][feat]


    def predict(self, x):
        return self.argmax(x)[0]

    def argmax(self, x):
        argmax = ()
        init = -float('inf')
        for label, w in self.w_vectors.iteritems():

            product = sum([x[feat] * w[feat] for feat in x])

            if product > init:
                argmax = (label, w)
                init = product
        return argmax

############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):

    def __init__(self, data):
        iterations = 20
        num_feat = max([len(exp[0]) for exp in data])
        examples = [({'x' + str((i+1)) : exp[0][i] for i in xrange(num_feat)}, exp[1]) for exp in data]

        self.classifier = MulticlassPerceptron(examples, iterations)


    def classify(self, instance):
        ins = {'x' + str(i + 1) : instance[i] for i in xrange(len(instance))}
        return self.classifier.predict(ins)

class DigitClassifier(object):

    def __init__(self, data):

        iterations = 12
        examples = self.format_data(data)

        self.classifier = MulticlassPerceptron(examples, iterations)

    def classify(self, instance):
        ins = {'x' + str(i + 1) : instance[i] for i in xrange(len(instance))}
        return self.classifier.predict(ins)

    def format_data(self, data):
        num_feat = 64  # hardcode assumption
        examples = []
        for exp in data:
            feat_map = dict()
            for i in xrange(num_feat):
                if exp[0][i] == 0:
                    continue
                else:
                    feat_map['x' + str(i + 1)] = exp[0][i]
            examples.append((feat_map, exp[1]))
        return examples

class BiasClassifier(object):

    def __init__(self, data):
        iterations = 10
        examples = self.format_data(data)
        self.classifier = BinaryPerceptron(examples, iterations)

    def classify(self, instance):
        ins = {'x1' : instance, 'x2' : 1}
        return self.classifier.predict(ins)

    def format_data(self, data):
        examples = [({'x1' : exp[0], 'x2' : 1}, exp[1]) for exp in data]

        return examples

class MysteryClassifier1(object):

    def __init__(self, data):
        iterations = 1
        examples = self.format_data(data)
        self.classifier = BinaryPerceptron(examples, iterations)

    def classify(self, instance):
#         ins = {'x' + str(i + 1) : instance[i] for i in xrange(len(instance))}
#         ins['x3'] = ins['x1']**2 + ins['x2']**2
#         ins['x4'] = 1
        ins = {'x1' : instance[0]**2 + instance[1]**2, 'x2' : 1}
        return self.classifier.predict(ins)

    def format_data(self, data):
#         examples = [({'x1' : exp[0][0], 'x2' : exp[0][1], 'x3' : exp[0][0]**2 + exp[0][1]**2, 'x4' : 1}, exp[1]) for exp in data]
        examples = [({'x1' : exp[0][0]**2 + exp[0][1]**2, 'x2' : 1}, exp[1]) for exp in data]
        return examples

class MysteryClassifier2(object):

    def __init__(self, data):
        iterations = 1
        examples = self.format_data(data)
        self.classifier = BinaryPerceptron(examples, iterations)

    def classify(self, instance):
        ins = {'x' + str(i + 1) : instance[i] for i in xrange(len(instance))}
        # add additional feature
        neg_cnt = 0
        for i in xrange(3):
            feat = 'x' + str(i + 1)
            if ins[feat] < 0:
                neg_cnt += 1
        ins['x4'] = neg_cnt % 2 - 0.5
        return self.classifier.predict(ins)

    def format_data(self, data):
        examples = [({'x' + str( i + 1) : exp[0][i] for i in xrange(len(exp[0]))}, exp[1]) for exp in data]
        # add in additional feature
        for exp in examples:
            neg_cnt = 0
            for i in xrange(3):
                feat = 'x' + str(i + 1)
                if exp[0][feat] < 0:
                    neg_cnt += 1
            exp[0]['x4'] = neg_cnt % 2 - 0.5
        return examples
