"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import math
from indexing import InvertedIndex
import numpy as np


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query
        query_tokens = self.tokenize(query)

        # 2. Fetch a list of possible documents from the index
        if self.stopwords:
            query_tokens = [token if not token in self.stopwords else None for token in query_tokens]
        

        possible_docs = set()
        for token in query_tokens:
            if token in self.index.vocabulary:  
                token_index = [new_index[0] for new_index in (self.index.get_postings(token))]
                possible_docs.update(token_index)

        possible_docs = list(possible_docs)
        doc_scores = []
        doc_to_term_count = defaultdict(Counter)
        query_count = Counter(query_tokens)

        for doc_id in set(query_tokens):
            if doc_id:
                doc_word_counts = self.index.get_postings(doc_id)
                for doc_word in doc_word_counts:
                    id= doc_word[0]
                    frequency = doc_word[1]
                    doc_to_term_count[id][doc_id] = frequency


        for docid, term in doc_to_term_count.items():
            result = self.scorer.score(docid, term, query_count)
            doc_scores.append((docid, result))
            # query_word_counts = {token: query_tokens.count(token) for token in query_tokens}  
            # doc_scores[doc_id] = self.scorer.score(doc_id[0],doc_word_counts, query_word_counts)

        
        # 4. Return **sorted** results as format [{docid: 100, score:0.5}, {docid: 10, score:0.2}]
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        # sorted_docs = [[doc_id, score] for doc_id, score in sorted_docs]
        # print(sorted_docs)

        return sorted_docs


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevant the document is for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        
        raise NotImplementedError
    


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        cosine_score = 0
        for q_term in query_word_counts:
            if not q_term or q_term not in doc_word_counts:
                continue
            cosine_score += query_word_counts[q_term] * doc_word_counts[q_term]

        # 2. Return the score

        return cosine_score
    

# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid)['length']
        mu = self.parameters['mu']

        # 2. Compute additional terms to use in algorithm
        # 3. For all query_parts, compute score
        score = 0 
        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                posting = self.index.get_postings(q_term)
                doc_tf = doc_word_counts[q_term] #document TF

                if doc_tf > 0:
                    query_tf = query_word_counts[q_term]
                    p_wc = sum([doc[1] for doc in posting]) / \
                        self.index.get_statistics()['total_token_count']
                    tfidf= np.log(1 + (doc_tf / (mu * p_wc)))

                    score += (query_tf * tfidf)

        score = score + len(query_word_counts) * np.log(mu / (doc_len + mu))

        # 4. Return the score
        return score


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
       
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        doc_length = self.index.get_doc_metadata(docid)['length']
        avg_doc_length = self.index.get_statistics()['mean_document_length']
        score = 0 
        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        for term, query_term_freq in query_word_counts.items():
            if term in doc_word_counts:  
                doc_term_freq = doc_word_counts[term]  

        # 3. For all query parts, compute the TF and IDF to get a score    
                metadata = self.index.get_term_metadata(term) 
                tf = ((self.k1 + 1) * doc_term_freq) / (self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length)) + doc_term_freq)
                idf = math.log((self.index.get_statistics()['number_of_documents'] - metadata['doc_frequency'] + 0.5) / (metadata['doc_frequency'] + 0.5))  
                score += idf * tf * ((self.k3 + 1) * query_term_freq) / (self.k3 + query_term_freq)
        
        # 4. Return score
        return score


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        doc_length = self.index.get_doc_metadata(docid)['length']
        avg_doc_length = self.index.get_statistics()['mean_document_length']
        score = 0

        # 2. Compute additional terms to use in algorithm
        for term, query_term_freq in query_word_counts.items():
            if term in doc_word_counts:
                doc_term_freq = doc_word_counts[term]
                freq = len(self.index.get_postings(term))
                tf = (1 + np.log(1 + np.log(doc_term_freq))) / (1 - self.b + (self.b * doc_length / avg_doc_length ))

                # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
                idf = np.log((self.index.get_statistics()['number_of_documents'] + 1) / (freq))
                score += query_term_freq * tf * idf

        # 4. Return the score
        return score
        return NotImplementedError


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        # doc_length = self.index.get_doc_metadata(docid)['length']
        total_docs = self.index.get_statistics()['number_of_documents']
        score = 0

        # 2. Compute additional terms to use in algorithm
        for term, query_term_freq in query_word_counts.items():
            if term in doc_word_counts:
                doc_term_freq = doc_word_counts[term]
                freq = len(self.index.get_postings(term))

                # 3. For all query parts, compute the TF and IDF to get a score
                idf = np.log((total_docs) / (freq)) +1
                tf = np.log(doc_term_freq + 1)
                score += tf * idf

        # 4. Return the score
        return score


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'lambda': 0.1}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Scores a document based on Jelinek-Mercer smoothing for a given query.
      
        Returns:
            The relevance score for the document given the query.
        """
        doc_len = self.index.get_doc_metadata(docid)['length']
        total_token_count = self.index.get_statistics()['total_token_count']
        lambda_param = self.parameters['lambda']

        score = 0.0

        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                posting = self.index.get_postings(q_term)  
                doc_tf = doc_word_counts.get(q_term, 0)  
                query_tf = query_word_counts[q_term]  

                # Corpus term frequency
                corpus_tf = sum([doc[1] for doc in posting])
                p_wc = corpus_tf / total_token_count 

                term_probability = (1 - lambda_param) * (doc_tf / doc_len) + lambda_param * p_wc

                if term_probability > 0:
                    score += query_tf * np.log(term_probability)

        return score

class SimpleWordMatchSearcher:
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        """
        Initialize the scorer with the index and optional parameters.
        """
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Calculates a simple score based on word matches between query and document.
        
        Args:
            docid: The ID of the document.
            doc_word_counts: Word counts in the document.
            query_word_counts: Word counts in the query.

        Returns:
            A float score representing the number of matching words.
        """
        match_score = 0
        for q_term in query_word_counts:
            if q_term in doc_word_counts:
                match_score += 1  # Increment score for each matching word found
        return match_score