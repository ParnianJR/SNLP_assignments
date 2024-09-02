from collections import Counter,defaultdict
from math import ceil, log2, prod, floor
import string
import math

def preprocess_text(text):
    #lowercasing
    text = ' '.join(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()

    return text
    raise NotImplementedError


class SmoothingCounter:
    def __init__(self, text,alpha=0):
        """ 
        :param text: preprocessed corpus
        :param d: discounting parameter, this is fixed
        :param alpha :  Smoothing factor for laplace smoothing
        :function kncounts(bigram) : Calculates the log probability of a bigram based on Kneser-Ney Counts
        :function logprob_alpha(bigram) : Calculates the log probabillty of a bigram based on Laplace smoothing
        """
        self.text = preprocess_text(text)
        self.d = 0.75 #according to the 2.3
        self.alpha = 0.5

        self.unigram_counts = Counter(self.text)
        self.bigram_counts = Counter(zip(self.text[:-1], self.text[1:]))
        self.trigram_counts = Counter(zip(self.text[:-2],self.text[1:-1],self.text[2:]))
        self.vocab = set(self.text)

        self.continuation_counts = defaultdict(int)
        for (w1, w2) in self.bigram_counts:
            self.continuation_counts[w2] += 1

        self.total_bigrams = sum(self.bigram_counts.values())
        self.total_trigrams = sum(self.trigram_counts.values())


    def knprob_bigram(self,bigram):
        '''returns the log probability of a bigram with counts adjusted for Knser-Ney Smoothing'''  
        w1, w2 = bigram
        bigram_count = self.bigram_counts[bigram]
        unigram_count_w1 = self.unigram_counts[w1]

        if bigram_count > 0:
            adjusted_bigram_count = max(bigram_count - self.d, 0)
            lambda_w1 = (self.d * len([w for w in self.vocab if (w1, w) in self.bigram_counts])) / unigram_count_w1
            continuation_prob_w2 = self.continuation_counts[w2] / self.total_bigrams
            prob = (adjusted_bigram_count / unigram_count_w1) + (lambda_w1 * continuation_prob_w2)
        else:
            prob = self.continuation_counts[w2] / self.total_bigrams

        return math.log2(prob) if prob > 0 else float('-inf')

        raise NotImplementedError

    def knprob_trigram(self,trigram):
        '''returns the log probability of a trigram with counts adjusted for Knser-Ney Smoothing'''  
        w1, w2, w3 = trigram
        trigram_count = self.trigram_counts[trigram]
        bigram_count_w1_w2 = self.bigram_counts[(w1, w2)]

        if trigram_count > 0:
            adjusted_trigram_count = max(trigram_count - self.d, 0)
            lambda_w1_w2 = (self.d * len([w for w in self.vocab if (w1, w2, w) in self.trigram_counts])) / bigram_count_w1_w2
            continuation_prob_w3 = self.continuation_counts[w3] / self.total_trigrams
            prob = (adjusted_trigram_count / bigram_count_w1_w2) + (lambda_w1_w2 * continuation_prob_w3)
        else:
            prob = math.exp(self.knprob_bigram((w2, w3)))

        return math.log2(prob) if prob > 0 else float('-inf')
        raise NotImplementedError


    def prob_alpha_bigram(self,bigram):
        '''returns the log probability of a bigram with counts adjusted for add-alpha Smoothing'''  
        w1, w2 = bigram
        bigram_count = self.bigram_counts[bigram]
        unigram_count_w1 = self.unigram_counts[w1]

        prob = (bigram_count + self.alpha) / (unigram_count_w1 + self.alpha * len(self.vocab))

        return math.log2(prob) if prob > 0 else float('-inf')
        raise NotImplementedError

    def prob_alpha_trigram(self, trigram):
        '''returns the probability of a trigram with counts adjusted for add-alpha Smoothing'''
        w1, w2, w3 = trigram
        trigram_count = self.trigram_counts[trigram]
        bigram_count_w1_w2 = self.bigram_counts[(w1, w2)]

        prob = (trigram_count + self.alpha) / (bigram_count_w1_w2 + self.alpha * len(self.vocab))

        return math.log2(prob) if prob > 0 else float('-inf')
        raise NotImplementedError