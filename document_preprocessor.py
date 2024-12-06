from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
#from nltk import word_tokenize
import re

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        """
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions if multiword_expressions else []

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Perform lower-casing and multi-word expression handling in a single pass.
        """
        if not input_tokens:
            return []

        #If no multi-word expressions, handle lowercasing only
        if not self.multiword_expressions:
            return [token.lower() if self.lowercase else token for token in input_tokens]
        mwe_tokenizer = MWETokenizer([tuple(mwe.split()) for mwe in self.multiword_expressions], separator='_')
        tokenized_text = mwe_tokenizer.tokenize(input_tokens)
        return [token.replace('_', ' ').lower() if self.lowercase else token.replace('_', ' ') for token in tokenized_text]

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.
        """
        tokens = text.split()
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = r'\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.
        """
        super().__init__(lowercase, multiword_expressions)
        #self.tokenizer = re.compile(token_regex)
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.
        """
        tokens = self.tokenizer.tokenize(text) 
        return self.postprocess(tokens) 


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words.
        """
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load("en_core_web_sm", disable=['parser'])

    def tokenize(self, text: str) -> list[str]:
        if not text:  # Handle empty string case
            return []

        # Handle multi-word expressions before Spacy tokenization
        for mwe in self.multiword_expressions:
            text = text.replace(mwe, mwe.replace(' ', '_'))

        # Tokenize using spaCy
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if '-' in token.text and not token.is_punct:
                parts = token.text.split('-')
                tokens.append(parts[0])
                tokens.append('-')
                tokens.append(parts[1])
            else:
                tokens.append(token.text)

        # Replace underscores in multi-word expressions back to spaces
        tokens = [token.replace('_', ' ') for token in tokens]
        return self.postprocess(tokens)

# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing

        # TODO (HW3): Create the dense tokenizer and query generation model using HuggingFace transformers
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name)
        self.model.to(self.device)

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                See the PDF for what you need to do for this part.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85

        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details

        # TODO (HW3): For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        # NOTE: Do not forget edge cases
        queries = []

        text = prefix_prompt + document
        input_ids = self.tokenizer.encode(text, max_length=document_max_token_length, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=50,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n_queries,
            num_beams=5,
            early_stopping=True

        )
        for output in outputs:
            queries.append(self.tokenizer.decode(output, skip_special_tokens=True))
        return queries
     


# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass


# python test_document_preprocessor_public.py 