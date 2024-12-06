'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
import gzip
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
import os
import json
from tqdm import tqdm 

class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents

        self.index = defaultdict(list)  # the index 

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        
        raise NotImplementedError
   

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """

        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.
        """
        doc_to_remove = []
        updated_documents = []

        for term, documents in self.index.items():
            for doc in documents:
                if doc != docid:
                    updated_documents.append(doc)
        
            if not updated_documents:
                doc_to_remove.append(term)
            else:
                self.index[term] = updated_documents

        for term in doc_to_remove:
            self.index[term] = None
            self.statistics['vocab'][term] = None

        if docid in self.document_metadata:
            del self.document_metadata[docid]


    def add_doc(self, docid: int, tokens: list[str], augmented_text: str = None, content: str = None) -> None:
        """
        Add a document to the index and update the index's metadata.

        Args:
            docid: The ID of the document.
            tokens: The tokens of the `augmented_text`.
            augmented_text: The processed text for indexing and ranking.
            content: The original text to display to the user.
        """
        token_frequencies = Counter(tokens)
        unique_tokens = 0

        for token, counts in token_frequencies.items():
            if token:
                self.statistics['vocab'][token] += counts
                self.vocabulary.add(token)
                self.index[token].append((docid, counts))
                unique_tokens += 1

        self.document_metadata[docid] = {
            "unique_tokens": unique_tokens,
            "length": len(tokens),
            "augmented_text": augmented_text,  # Store augmented_text
            "content": content  # Store content
        }


    def get_document(self, docid: int) -> str:
        """
        Retrieve the text of a document given its ID.

        Args:
            docid: The ID of the document.

        Returns:
            The text of the document or a default message if not available.
        """
        return self.document_metadata.get(docid, {}).get('content', 'Text not available')

    def get_postings(self, term: str) -> list:
        # return self.index.get(term, [])
        if term in self.index:
            return list(self.index.get(term))
        else:
            return []
        # raise NotImplementedError
        

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        
        if doc_id not in self.document_metadata:
            return {"unique_tokens": 0, "length": 0} 
        
        doc_data = self.document_metadata[doc_id]

        unique_tokens = len(set([token for token in doc_data if token is not None]))
        length = len([token for token in doc_data if token is not None])

        return self.document_metadata[doc_id] 
        # raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        
        if term in self.index:
            term_count = sum([t_count for _, t_count in self.index[term]])
            doc_frequency = len(self.index[term])
            return {"term_count": term_count, 
                    "doc_frequency": doc_frequency}
        else:
            return {"term_count": 0, "doc_frequency": 0}
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
       
        unique_token_count = len(self.statistics['vocab'])
        total_token_count =sum(self.document_metadata[docid]["length"] for docid in self.document_metadata)
        number_of_documents = len(self.document_metadata)
        mean_document_length =total_token_count / number_of_documents if number_of_documents else 0

        return {
            "unique_token_count": unique_token_count,
            "total_token_count": total_token_count,
            "number_of_documents": number_of_documents,
            "mean_document_length": mean_document_length
        }

    def save(self, index_directory_name = 'idn') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """

        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)

        index_file_path = index_directory_name +  "/index.json"

        with open(index_file_path, 'w', encoding='utf-8') as f:
            json.dump({"index": dict(self.index), "metadata": self.document_metadata}, f)


    def load(self, index_directory_name = 'idn') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        index_file = os.path.join(index_directory_name, "index.json")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            data= json.load(f)
            self.index = defaultdict(list, data['index'])
            self.document_metadata = data['metadata']
    

class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
       
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata based on this document's condition.
        """

        self.document_metadata[docid] = {
            "unique_tokens": len(set(tokens)),
            "length": len(tokens)
        }
 
        for pos, token in enumerate(tokens):
            if not token:
                continue
            self.index[token].append((docid, pos))

        self.statistics['vocab'].update(tokens)
        self.vocabulary.update(tokens)

        
    def get_postings(self, term: str) -> list:

        postings = defaultdict(list)
        if term in self.index:
            for doc_id, pos in self.index.get(term, []):
                postings[doc_id].append(pos)
            return [(doc_id, len(positions), positions) for doc_id, positions in postings.items()]
        else:
            return []
    

       
    

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    def create_index(index_type: IndexType, dataset_path: str,
                    document_preprocessor: Tokenizer, stopwords: set[str],
                    minimum_word_frequency: int, augmented_text_key="augmented_text", content_key="content",
                    max_docs: int = 1000) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.

        Returns:
            An inverted index
        
        '''
        # first count each term by split the term in the document and lowercasing and then tokenize each term 
        # minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
#                 If the token does not appear in the document at least for the set frequency, it will not be indexed.
#                 Setting a value of 0 will completely ignore the parameter. use split by whitespace and then lowercased to count each term
        if index_type == IndexType.BasicInvertedIndex:
            inverted_index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            inverted_index = PositionalInvertedIndex()
        else:
            raise ValueError(f"Index type {index_type} not supported")

        doc_count = 0
        term_count = 0
        token_count = Counter()
        
        open_data = gzip.open if dataset_path.endswith(".gz") else open
        with open_data(dataset_path, 'rt', encoding='utf-8') as f:
            for doc_id, line in tqdm(enumerate(f, start=1)):
                if max_docs >= 0 and doc_count >= max_docs:
                    break

                document = json.loads(line)
                doc_id = document.get('docid', doc_id)
                augmented_text = document.get(augmented_text_key, "")
                content = document.get(content_key, "")
                token_word = document_preprocessor.tokenize(augmented_text)

                if stopwords:
                    token_word = [token if token not in stopwords else None for token in token_word]

                inverted_index.add_doc(doc_id, token_word, augmented_text=augmented_text, content=content)
                doc_count += 1

        return inverted_index


    @staticmethod
    def new_method():
        index_type = IndexType.BasicInvertedIndex







# TODO for each inverted index implementation, use the Indexer to create an index with the first 10, 100, 1000, and 10000 documents in the collection (what was just preprocessed). At each size, record (1) how
# long it took to index that many documents and (2) using the get memory footprint function provided, how much memory the index consumes. Record these sizes and timestamps. Make
# a plot for each, showing the number of documents on the x-axis and either time or memory
# on the y-axis.

'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1
    
    def save(self):
        print('Index saved!')

if __name__ == '__main__':

    # from document_preprocessor import RegexTokenizer
    
    # file_path = '../data/wikipedia_200k_dataset.jsonl.gz'
    # tokenizer = RegexTokenizer()
   
    pass