import math
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import List, Union


class LanguageModel:
    def __init__(self, n_gram: int = 1) -> None:
        self.n_gram = n_gram # Keep track of the n-gram value
        self.probs = {} # To store the probabilities of the n-grams i.e. {tuple(w1, w2, .., wn): probability, ...}
        self.n_gram_counts = {} # To store the counts of the n-grams i.e. {tuple(w1, w2, .., wn): count, ...}
        self.n_minus_1_gram_counts = {} # To store the counts of the n-1 grams i.e. {tuple(w1, w2, .., wn-1): count, ...}
        self.unigram_counts = {} # To store the counts of the unigrams i.e. {tuple(w,): count, ...}

    def get_counts(self, tokens: List[str], n: int = 1) -> dict:
        """
        Takes a list of tokens and returns the n-gram counts.

        params:
        - tokens: A list of tokens
        - n: The value of n for n-grams

        returns a dictionary of n-gram counts i.e. {(w1, w2, .., wn): count, ...}

        Example:
        tokens = ['this', 'is', 'a', 'sentence']
        get_counts(tokens, n=2)
        Output: {('this', 'is'): 1, ('is', 'a'): 1, ('a', 'sentence'): 1}
        """
        n_gram_counts = defaultdict(int)

        # ====================================
        # Your code here
        for i in range(len(tokens)-n+1):
            n_gram = tuple(tokens[i:i + n])
            n_gram_counts[n_gram] += 1


        return n_gram_counts
        # ====================================
        raise NotImplementedError

    def train(self, tokens: List[str]) -> None:
        """
        Takes a text and trains the language model.
        Training here means computing the probabilities of the n-grams and storing them in self.probs.

        params:
        - tokens: A list of tokens

        returns None
        """

        self.n_gram_counts = self.get_counts(tokens, n=self.n_gram)  # N(h, w)
        self.n_minus_1_gram_counts = self.get_counts(tokens, n=self.n_gram - 1)  # N(h)
        self.unigram_counts = self.get_counts(tokens, n=1)  # N(w)

        # ====================================
        # Your code here

        for n_gram, count in self.n_gram_counts.items():
            hist = n_gram[:-1] #all elements but the last
            self.probs[n_gram] = count / self.n_minus_1_gram_counts[hist] if hist in self.n_minus_1_gram_counts else 0.0

        # ====================================
        #raise NotImplementedError

    def generate(self, history_tokens: List[str]) -> str:
        """
        Takes a list of tokens and returns the most likely next token.
        Return None if the history is not present in the model.

        params:
        - history_tokens: A list of tokens

        returns the next token
        """

        # Convert it into a tuple, in case it's already not
        history_tokens = tuple(history_tokens)

        if len(history_tokens) != self.n_gram - 1:
            # If history is longer than what's required for our n-gram model
            # simply take the last n-1 tokens
            history_tokens = history_tokens[-(self.n_gram - 1) :]

        max_prob = 0
        next_token = None

        # ====================================
        # Your code here
        for n_gram, prob in self.probs.items():
            if n_gram[:-1] == history_tokens:
                if prob > max_prob:
                    max_prob = prob
                    next_token = n_gram[-1]

        # ====================================

        return next_token

    def get_smoothed_probs(self, n_gram: List[str], d: float = 0.1) -> float:
        """
        Takes a n-gram and returns the smoothed probability using absolute discounting.

        params:
        - n_gram: A list/tuple of tokens (w1, w2, .., wn)
        - d: The discounting factor

        returns the smoothed probability
        """

        n_gram = tuple(n_gram)
        history = n_gram[:-1]
        w = n_gram[-1]

        # ====================================
        # Your code here
        # Compute step by step to prevent errors
        N_1_plus = len(self.unigram_counts)

        history_counts = self.get_counts(history, 0)
        history_count = int(list(history_counts.values())[0])
        #print(history_count)
        n_gram_counts = self.get_counts(n_gram, 0)
        n_gram_count = int(list(n_gram_counts.values())[0])
        count_unique_follow_w = sum(1 for ngram in n_gram_counts if ngram[:-1] == history)

        sum_unigram_counts = sum(self.unigram_counts.values())
        # Computing the lambda(.) value
        
        lambda_ = (d / sum_unigram_counts * N_1_plus)

        # Computing the lambda(w_{i-1}) value
        
        lambda_h = count_unique_follow_w * d / history_count if history_count != 0 else 0.0

        # Computing P_abs(w_i)

        P_abs_w_i = max(self.unigram_counts.get((w,), 0) - d, 0) / sum_unigram_counts + (d / sum_unigram_counts) * len(self.unigram_counts)

        # Computing P_abs(w_i | w_{i-1})
        P_abs_w_i_given_h = max(n_gram_count - d, 0) / history_count + lambda_h * P_abs_w_i

        # ====================================

        return P_abs_w_i_given_h

    def generate_absolute_smoothing(self, history_tokens: List[str], d: float = 0.1) -> str:
        """
        Takes a list of tokens and returns the most likely next token using absolute discounting.

        params:
        - history_tokens: A list of tokens (w1, w2, .., wn-1) for which we want to predict the next token
        - d: The discounting factor

        returns the next token
        """

        # Convert it into a tuple, in case it's already not
        history_tokens = tuple(history_tokens)

        if len(history_tokens) != self.n_gram - 1:
            # If history is longer than what's required for our n-gram model
            # simply take the last n-1 tokens
            history_tokens = history_tokens[-(self.n_gram - 1) :]

        max_prob = 0
        next_token = None

        # ====================================
        # Your code here
        for n_gram in self.n_gram_counts:
            if n_gram[:-1] == history_tokens:
                prob = self.get_smoothed_probs(list(n_gram), d)
                if prob > max_prob:
                    max_prob = prob
                    next_token = n_gram[-1]
        # ====================================

        return next_token

    def perplexity(self, tokens: List[str], n_words: int, d: float = None) -> float:
        """
        Takes a list of tokens and returns the perplexity of the language model (per word)
        Remember to normalize the probabilities by the number of words (not number of tokens or n-grams, see formula) in the text.

        params:
        - tokens: A list of tokens
        - n_words: The number of words in the text
        - d: The discounting factor for absolute discounting (only for absolute discounting, otherwise None)

        returns the perplexity of the language model
        """
        log_prob = 0
        ngrams = self.get_counts(tokens, n=self.n_gram).keys()

        for ngram in ngrams:
            if d is None:
                prob = self.probs.get(ngram, 1e-10)  # Use a small probability for unseen n-grams
            else:
                prob = self.get_smoothed_probs(list(ngram), d)
            
            log_prob += math.log(prob)

        perplexity = math.exp(-log_prob / n_words)
        return perplexity
        # ====================================
        raise NotImplementedError