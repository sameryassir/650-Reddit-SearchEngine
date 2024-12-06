import math
import numpy as np
import pandas as pd
from ranker import RelevanceScorer, BM25, InvertedIndex
import csv


"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 20) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """

    search_result_relevances = search_result_relevances[:cut_off]
    num_relevant = 0
    precision_at_k = []

    for i, rel in enumerate(search_result_relevances):
        if rel == 1:
            num_relevant += 1
            precision_at_k.append(num_relevant / (i + 1))

    if not precision_at_k:
        return 0.0

    return sum(precision_at_k) / len(precision_at_k)


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_of: int = 20):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    relevances = np.array(search_result_relevances[:cut_of])
    discounts = np.log2(np.arange(2, len(relevances) + 2))  # log2(i+1) for i starting at 1
    dcg = np.sum(relevances / discounts)

    ideal_relevances = np.array(ideal_relevance_score_ordering[:cut_of])
    ideal_dcg = np.sum(ideal_relevances / discounts)

    if ideal_dcg == 0:
        return 0.0

    ndcg_score = dcg / ideal_dcg
    # Return NDCG
    return ndcg_score
    pass


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset
    relevance_data = []
    with open(relevance_data_filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            relevance_data.append(row)

    queries = list(set(row['query'] for row in relevance_data))
    relevance_dict = {query: [] for query in queries}
    for row in relevance_data:
        relevance_dict[row['query']].append(row)
    # relevance_data = pd.read_csv(relevance_data_filename)
    
    # queries = relevance_data['query'].unique()
    queries = list(set(row['query'] for row in relevance_data))

    map_scores = []
    ndcg_scores = []

    # TODO: Run each of the dataset's queries through your ranking function
    for query in queries:
        query_relevances = [row for row in relevance_data if row['query'] == query]
        # query_relevances = relevance_data[relevance_data['query'] == query]
        # search_results = ranker.query(query)
    
    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

        # binary_relevances = [1 if rel >= 4 else 0 for rel in query_relevances['rel']]
        binary_relevances = [1 if int(row['rel']) >= 4 else 0 for row in query_relevances]
        map_scores.append(map_score(binary_relevances))

        # Use original relevance scores for NDCG calculation
        # ndcg_scores.append(ndcg_score(query_relevances['annotation_rel_score'], sorted(query_relevances['rel'], reverse=True)))
        rel_scores = [int(row['rel']) for row in query_relevances]
        ideal_rel_scores = sorted(rel_scores, reverse=True)
        ndcg_scores.append(ndcg_score(rel_scores, ideal_rel_scores))

    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    return {'map': avg_map, 'ndcg': avg_ndcg, 'map_list': map_scores, 'ndcg_list': ndcg_scores}
    
    return {'map': 0, 'ndcg': 0, 'map_list': [], 'ndcg_list': []}

# def run_relevance_tests(relevance_data_filename: str, ranker) -> dict:
#     relevance_data = pd.read_csv(relevance_data_filename)
    
#     # Check if it's a DataFrame
#     if not isinstance(relevance_data, pd.DataFrame):
#         raise ValueError("Expected relevance_data to be a DataFrame.")
    
#     # Extract unique queries
#     queries = relevance_data['query'].unique()
#     relevance_dict = {query: relevance_data[relevance_data['query'] == query] for query in queries}

#     map_scores = []
#     ndcg_scores = []

#     for query in queries:
#         query_relevances = relevance_dict[query]
#         search_results = ranker.query(query)  # Assuming the ranker has a .query() method

#         # Convert annotation scores to binary relevance scores for MAP calculation
#         binary_relevances = [1 if rel >= 4 else 0 for rel in query_relevances['annotation_rel_score']]

#         # Calculate MAP and NDCG (ensure your map and ndcg functions are defined and accept necessary arguments)
#         map_score_value = map_score(binary_relevances, search_results)
#         ndcg_score_value = ndcg_score(query_relevances['annotation_rel_score'], search_results)
        
#         map_scores.append(map_score_value)
#         ndcg_scores.append(ndcg_score_value)

#     avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
#     avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

#     return {'map': avg_map, 'ndcg': avg_ndcg, 'map_list': map_scores, 'ndcg_list': ndcg_scores}


if __name__ == '__main__':
    # from pipeline import initialize
    # search_engine = initialize()
       # run_relevance_tests(search_engine)

    # relevance_data_filename = 'data/BM25_baseline_output_rel_score.csv'
    # bm25_ranker = BM25(InvertedIndex)
    # run_relevance_tests(relevance_data_filename, bm25_ranker)

    relevance_data_filename = 'data/BM25_baseline_output_rel_score.csv'
    bm25_ranker = BM25(InvertedIndex)  # Ensure InvertedIndex is initialized properly
    scores = run_relevance_tests(relevance_data_filename, bm25_ranker)
    
    # Print the average MAP and NDCG scores
    print(f"Average MAP Score: {scores['map']:.4f}")
    print(f"Average NDCG Score: {scores['ndcg']:.4f}")


    pass
