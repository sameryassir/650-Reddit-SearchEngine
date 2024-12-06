from tqdm import tqdm
from keybert import KeyBERT
from transformers import T5Tokenizer
import jsonlines

class DocumentAugmentor:
    '''
    Augments documents by generating keywords.
    '''
    def __init__(self, doc_to_post: dict[int, str], stopwords: str = 'english', ngram_range: tuple[int] = (1, 2)):
        self.doc_to_post = doc_to_post
        self.model = KeyBERT()
        self.ngram_range = ngram_range
        self.stopwords = stopwords
        
        # Initialize a T5 Tokenizer (unused in this code but retained from the original model)
        self.stokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

    def augment_docs(self, docids: list[int]) -> dict[int, str]:
        '''
        Augments documents by adding keywords.

        Args:
            docids: List of document IDs to process.

        Returns:
            A dictionary mapping document IDs to augmented text (original + keywords).
        '''
        posts = [self.doc_to_post[docid] for docid in docids]
        posts_to_keywords = self.model.extract_keywords(
            posts, 
            keyphrase_ngram_range=self.ngram_range, 
            stop_words=self.stopwords, 
            use_mmr=True
        )

        augmented_text = {}
        for i, keyword_list in tqdm(enumerate(posts_to_keywords), total=len(posts_to_keywords), desc="Augmenting documents"):
            docid = docids[i]
            post = posts[i]
            
            # Generate keywords
            keywords = [keyword for keyword, _ in keyword_list]
            keyword_text = " ".join(keywords)
            
            # Combine original post and keywords
            augmented_text[docid] = f"{post} {keyword_text}"
        
        return augmented_text

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a dictionary with docid as keys and content as values.
    """
    data = {}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data[int(obj['docid'])] = obj['content']
    return data

def write_jsonl(data, output_file):
    """
    Writes the augmented data into a JSONL file.
    """
    with jsonlines.open(output_file, mode='w') as writer:
        for docid, augmented_text in data.items():
            writer.write({"docid": docid, "augmented_text": augmented_text})
    print(f"Augmented data has been written to {output_file}")

def main():
    input_file = '../data/documents_full_contents.jsonl'
    doc_to_post = read_jsonl(input_file)

    augmentor = DocumentAugmentor(doc_to_post)

    docids = list(doc_to_post.keys())
    augmented_documents = augmentor.augment_docs(docids)

    output_file = '../data/augmented_data.jsonl'
    write_jsonl(augmented_documents, output_file)
    print("Done!")

if __name__ == '__main__':
    main()
