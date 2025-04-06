from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the retrieval evaluator with an embedding model
        
        :param embedding_model: Name of the embedding model to use
        """
        logger.info(f"Initializing RetrievalEvaluator with model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        :param texts: List of text strings to embed
        :return: Numpy array of embeddings
        """
        logger.info(f"Embedding {len(texts)} texts")
        return self.embedding_model.encode(texts)

    def retrieve_top_k_chunks(self,
                              query: str,
                              corpus_chunks: List[Dict],
                              k: int = 5) -> List[Dict]:
        """
        Retrieve top K most similar chunks to a query
        
        :param query: Query string
        :param corpus_chunks: List of dictionaries containing content and metadata
        :param k: Number of top chunks to retrieve
        :return: List of top K most similar chunks with metadata
        """
        logger.info(f"Retrieving top {k} chunks for query: '{query[:50]}...'")
        
        # Extract content for embedding
        chunk_contents = [chunk["content"] for chunk in corpus_chunks]

        # Embed query and corpus chunks
        query_embedding = self.embed_texts([query])[0]
        chunk_embeddings = self.embed_texts(chunk_contents)

        # Compute cosine similarities
        similarities = cosine_similarity(
            [query_embedding], chunk_embeddings)[0]

        # Get indices of top K chunks
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top chunks with their similarity scores
        result = []
        for idx in top_k_indices:
            chunk = corpus_chunks[idx].copy()
            chunk["similarity_score"] = float(similarities[idx])
            result.append(chunk)

        return result

    @staticmethod
    def intersect_ranges(range1: Tuple[int, int], range2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Calculate the intersection between two ranges
        
        :param range1: First range as (start, end)
        :param range2: Second range as (start, end)
        :return: Intersection range or None if no intersection
        """
        intersect_start = max(range1[0], range2[0])
        intersect_end = min(range1[1], range2[1])

        if intersect_start <= intersect_end:
            return (intersect_start, intersect_end)
        else:
            return None

    @staticmethod
    def sum_of_ranges(ranges: List[Tuple[int, int]]) -> int:
        """
        Calculate sum of lengths of all ranges
        
        :param ranges: List of ranges as (start, end) tuples
        :return: Sum of range lengths
        """
        return sum(end - start for start, end in ranges)

    @staticmethod
    def union_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping ranges
        
        :param ranges: List of ranges as (start, end) tuples
        :return: List of merged ranges
        """
        if not ranges:
            return []

        # Sort ranges based on starting index
        sorted_ranges = sorted(ranges, key=lambda x: x[0])

        # Initialize with the first range
        merged_ranges = [sorted_ranges[0]]

        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged_ranges[-1]

            # Check if current range overlaps or is contiguous with last range
            if current_start <= last_end:
                # Merge the ranges
                merged_ranges[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as new range
                merged_ranges.append((current_start, current_end))

        return merged_ranges

    def calculate_precision_recall(self,
                                   retrieved_chunks: List[Dict],
                                   reference_ranges: List[Dict],
                                   ) -> Tuple[float, float]:
        """
        Calculate precision and recall based on text ranges
        
        :param retrieved_chunks: List of retrieved chunks with metadata
        :param reference_ranges: List of reference ranges with metadata
        :return: Tuple of (precision, recall) scores
        """
        numerator_sets = []

        # Convert reference chunks to range tuples
        reference_range_tuples = [
            (ref["start_index"], ref["end_index"]) for ref in reference_ranges]

        for chunk in retrieved_chunks:
            # Unpack chunk start and end indices
            chunk_start, chunk_end = chunk['start_index'], chunk['end_index']

            for ref_start, ref_end in reference_range_tuples:
                # Calculate intersection between chunk and reference
                intersection = self.intersect_ranges(
                    (chunk_start, chunk_end), (ref_start, ref_end))

                if intersection is not None:
                    # Add intersection to numerator sets
                    numerator_sets = self.union_ranges(
                        [intersection] + numerator_sets)

        # Calculate values
        if numerator_sets:
            numerator_value = self.sum_of_ranges(numerator_sets)
        else:
            numerator_value = 0

        # Calculate denominators
        recall_denominator = self.sum_of_ranges(reference_range_tuples)
        precision_denominator = self.sum_of_ranges(
            [(chunk['start_index'], chunk['end_index']) for chunk in retrieved_chunks])

        # Calculate scores
        recall = numerator_value / recall_denominator if recall_denominator > 0 else 0
        precision = numerator_value / precision_denominator if precision_denominator > 0 else 0

        return precision, recall

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall
        
        :param precision: Precision value
        :param recall: Recall value
        :return: F1 score
        """
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)