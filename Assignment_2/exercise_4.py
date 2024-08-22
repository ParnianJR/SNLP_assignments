import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM


class GPT2Model:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.eval() # eval mode since we don't need to train

    def forward(self, text):
        """
        Take a text and return the probabilities of the next token
        """

        # your code here
        # Tokenize and encode the text
        inputs = self.tokenizer(text, return_tensors='pt')

        # Get the logits from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the logits of the last token
        logits = outputs.logits[:, -1, :]
        
        # Apply softmax to get the probabilities
        exp_logits = torch.exp(logits - torch.max(logits))
        prob_dist = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        
        return prob_dist

        raise NotImplementedError

    def decode(self, indices):
        return self.tokenizer.decode(indices)

    def greedy_sample(self, text):
        """
        Takes a context and greedily returns the most likely next token
        """

        # your code here

        prob_dist = self.forward(text)
        next_token = torch.argmax(prob_dist, dim=-1).item()
        
        return self.decode(next_token)

    def random_sample(self, text):

        # your code here
        prob_dist = self.forward(text)
        next_token = torch.multinomial(prob_dist, num_samples=1).item()

        return self.decode(next_token)

    def rejection_sample(self, text):

        # your code here
        while True:
            # Generate a random number
            r = torch.rand(1).item()

            # Get probability distribution
            prob_dist = self.forward(text)

            # Randomly sample a token from the output probability
            next_token = torch.multinomial(prob_dist, num_samples=1).item()

            # Check if r < p
            if r < prob_dist[0][next_token]:
                # Accept the token
                return self.decode(next_token)
