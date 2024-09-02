from collections import defaultdict
import math

class CountTree():
    def __init__(self, n=4):
        if n == 1:
            self.nodes = defaultdict(int)
        else:
            self.nodes = defaultdict(lambda: CountTree(n=n-1))

    def add(self, ngram):
        if len(ngram) == 1:
            self.nodes[ngram[0]] += 1
        else:
            self.nodes[ngram[-1]].add(ngram[:-1])

    def get(self, ngram):
        if len(ngram) == 0:
            total = 0
            for value in self.nodes.values():
                if isinstance(value, CountTree):
                    total += value.get(())
                else:
                    total += value
            return total
        else:
            next_node = self.nodes.get(ngram[-1])
            if next_node is None:
                return 0
            elif isinstance(next_node, CountTree):
                return next_node.get(ngram[:-1])
            else:
                return next_node

    def cond_prob(self, word, history):
        history_count = self.get(history)
        if history_count == 0:
            return 0
        ngram = history + (word,)
        ngram_count = self.get(ngram)
        return ngram_count / history_count

    def perplexity(self, ngrams, vocab):
        total_log_prob = 0
        N = len(ngrams)
        for ngram in ngrams:
            history = ngram[:-1]
            word = ngram[-1]
            p4 = self.cond_prob(word, history)
            p0 = 1 / vocab
            p = 0.75 * p4 + 0.25 * p0
            total_log_prob += math.log(p)
        return math.exp(-total_log_prob / N)

    def prune(self, k):
        def prune_node(node):
            if isinstance(node, CountTree):
                sum_counts = sum(prune_node(child) for child in node.nodes.values())
                if sum_counts <= k:
                    return sum_counts
                return sum_counts
            return node

        keys_to_remove = []
        for key, value in self.nodes.items():
            if isinstance(value, CountTree):
                sum_counts = prune_node(value)
                if sum_counts <= k:
                    keys_to_remove.append(key)
            elif value <= k:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if isinstance(self.nodes[key], CountTree):
                self.nodes[key] = sum(prune_node(child) for child in self.nodes[key].nodes.values())
            else:
                del self.nodes[key]



    