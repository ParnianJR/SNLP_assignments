import math
from collections import Counter
from typing import List, Union
import morfessor
from tokenizers import Tokenizer
from morfessor.baseline import BaselineModel


class TokenizerEntropy:
    def tokenize_bpe(self, tokenizer: Tokenizer, text: str) -> List[str]:
        """
        Takes the BPE tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained BPE tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """
        # ====================================
        # Your code here
        text = text.lower()

        tokens = tokenizer.encode(text).tokens
        return tokens
        # ====================================
        raise NotImplementedError

    def tokenize_morfessor(self, tokenizer: BaselineModel, text: str) -> List[str]:
        """
        Takes the Morfessor tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained Morfessor tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """
        # ====================================
        # Your code here
        text = text.lower()

        tokens = tokenizer.viterbi_segment(text)[0]
        return tokens
        # ====================================
        raise NotImplementedError

    def get_probs(self, tokens: List[str]):
        """
        Takes a list of tokens and compute the probability distribution of the tokens.

        params:
        - tokens: A list of tokens

        returns a dictionary of token probabilities i.e. {token: probability, ...}
        """
        # ====================================
        # Your code here
        counts = Counter(tokens)
        total_count = sum(counts.values())
        prob_dict = {token:(count/total_count) for token,count in counts.items()}
        #sorted_prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
        #print(f"Token Probabilities: {sorted_prob_dict}")
        return prob_dict
        # ====================================
        raise NotImplementedError

    def compute_entropy(
        self, text: str, tokenizer: Union[Tokenizer, BaselineModel]
    ) -> float:
        """
        Takes the input text and the tokenizer and returns the entropy of the text.

        params:
        - text: The input text
        - tokenizer: The pre-trained tokenizer (BPE or Morfessor)

        returns the entropy of the text
        """
        tokens = None
        # tokenize the input text
        if isinstance(tokenizer, Tokenizer):
            tokens = self.tokenize_bpe(tokenizer, text)
        elif isinstance(tokenizer, BaselineModel):
            tokens = self.tokenize_morfessor(tokenizer, text)
        else:
            raise ValueError("Tokenizer not supported.")

        # ====================================
        # Your code here
        # get the probabilities of each token
        token_probs = self.get_probs(tokens)
        sorted_token_probs = sorted_prob_dict = dict(sorted(token_probs.items(), key=lambda item: item[1], reverse=True))
        print("sorted_token_probs: ",sorted_token_probs)
        # Compute the entropy
        entropy = 0
        entropy = -sum(p * math.log2(p) for p in token_probs.values())
        return entropy
        # ====================================
        raise NotImplementedError
