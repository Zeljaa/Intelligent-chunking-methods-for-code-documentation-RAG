import pandas as pd
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

import re
import logging



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import the base chunker class
from base_chunker import BaseChunker

# Import the fixed token chunker 
from fixed_token_chunker import FixedTokenChunker

from RetrievalEvaluator import RetrievalEvaluator



def process_chat_logs(file_path: str, chunker: BaseChunker) -> List[Dict[str, Any]]:
    """
    Process chat logs and split them into chunks
    
    :param file_path: Path to the chat log file
    :param chunker: Chunker instance to use for splitting text
    :return: List of chunks with their positions in the original file
    """
    logger.info(f"Processing chat logs from: {file_path}")
    
    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = file.read().strip().replace("\"", "'")
    
    # Find all lists using regex
    list_matches = re.findall(r'\[.*?\]', raw_data, re.DOTALL)
    
    all_chunks = []
    file_offset = 0
    
    # Process each list
    for lst_str in list_matches:
        try:
            # Find where this list starts in the raw file
            list_start_idx = raw_data.find(lst_str, file_offset)
            file_offset = list_start_idx + len(lst_str)
            
            # Extract individual dictionaries
            dict_matches = re.findall(r'\{.*?\}', lst_str, re.DOTALL)
            
            for dict_str in dict_matches:
                # Find where this dictionary starts in the raw file
                dict_start_idx = raw_data.find(dict_str, list_start_idx)
                
                # Extract content field
                content_match = re.search(r'\'content\': \'(.*?)\', \'role\'', dict_str, re.DOTALL)
                if content_match:
                    content = content_match.group(1)
                    content_start_idx = dict_str.find(content)
                    
                    # Calculate absolute position in the raw file
                    abs_content_start_idx = dict_start_idx + content_start_idx
                    
                    # Process content with the chunker
                    chunks = chunker.split_text(content)
                    
                    # Adjust indices to be relative to the raw file
                    for chunk in chunks:
                        chunk_start_index = content.find(chunk)
                        all_chunks.append({
                            "content": chunk,
                            "start_index": abs_content_start_idx + chunk_start_index,
                            "end_index": abs_content_start_idx + chunk_start_index + len(chunk)
                        })
        
        except Exception as e:
            logger.error(f"Error processing list: {e}")
    
    logger.info(f"Extracted {len(all_chunks)} chunks from the chat logs")
    return all_chunks

def process_chat_logs_whole(file_path: str, chunker: BaseChunker) -> List[Dict[str, Any]]:
    """
    Process chat logs by chunking the entire raw file
    
    :param file_path: Path to the chat log file
    :param chunker: Chunker instance to use for splitting text
    :return: List of chunks with their positions in the original file
    """
    logger.info(f"Processing chat logs from: {file_path}")
    
    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = file.read().strip().replace("\"", "'")
    
    # Process the entire raw data with the chunker
    chunks = chunker.split_text(raw_data)
    
    # Create list to store chunks with their positions
    all_chunks = []
    
    # For each chunk, find its position in the original file
    for chunk in chunks:
        # Find the start index of this chunk in the raw data
        # Note: This will find the first occurrence if the chunk appears multiple times
        chunk_start_index = raw_data.find(chunk)
        
        if chunk_start_index >= 0:  # Check if the chunk was found
            all_chunks.append({
                "content": chunk,
                "start_index": chunk_start_index,
                "end_index": chunk_start_index + len(chunk)
            })
        else:
            logger.warning(f"Could not find chunk in raw data: {chunk[:50]}...")
    
    logger.info(f"Extracted {len(all_chunks)} chunks from the chat logs")
    return all_chunks


def load_evaluation_questions(file_path: str) -> List[Dict]:
    """
    Load evaluation questions and their reference ranges
    
    :param file_path: Path to the evaluation questions JSON file
    :return: List of questions with their reference ranges
    """
    logger.info(f"Loading evaluation questions from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    for question in questions:
        question["references"] = json.loads(question["references"])

    logger.info(f"Loaded {len(questions)} evaluation questions")
    return questions

def run_evaluation_pipeline(
    corpus_file: str,
    questions_file: str,
    corpus_solo_content: bool = False,
    chunker_type: str = "fixed_token",
    chunk_sizes: List[int] = [50, 100, 200],
    overlap_percent: float = 0.15,  # Overlap as a percentage of chunk size
    embedding_model: str = "all-MiniLM-L6-v2",
    top_k_values: List[int] = [1, 3, 5, 10],
    output_file: str = "evaluation_results.json"
) -> Dict:
    """
    Run the complete retrieval evaluation pipeline with multiple chunk sizes and k values
    
    :param corpus_file: Path to the corpus file
    :param questions_file: Path to the questions file
    :param chunker_type: Type of chunker to use ('fixed_token' or other future implementations)
    :param chunk_sizes: List of chunk sizes to evaluate
    :param overlap_percent: Overlap between chunks as a percentage of chunk size
    :param embedding_model: Name of the embedding model to use
    :param top_k_values: List of k values to evaluate
    :param output_file: Path to the output file
    :return: Dictionary of evaluation results
    """
    logger.info("Starting retrieval evaluation pipeline")
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(embedding_model=embedding_model)
    
    # Load evaluation questions
    questions = load_evaluation_questions(questions_file)
    
    # Sort k values to find max_k
    sorted_k_values = sorted(top_k_values)
    max_k = sorted_k_values[-1]
    
    # Initialize results structure
    results = {
        "metadata": {
            "corpus_file": corpus_file,
            "questions_file": questions_file,
            "chunker_type": chunker_type,
            "chunk_sizes": chunk_sizes,
            "overlap_percent": overlap_percent,
            "embedding_model": embedding_model,
            "top_k_values": top_k_values,
            "num_questions": len(questions)
        },
        "by_chunk_size": {},
        "by_question": {},
        "summary": {}
    }
    
    # Process each chunk size
    for chunk_size in chunk_sizes:
        chunk_overlap = int(chunk_size * overlap_percent)
        logger.info(f"Processing with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Initialize chunker for this size
        if chunker_type == "fixed_token":
            chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
        
        # Process corpus into chunks
        if corpus_solo_content:
            corpus_chunks = process_chat_logs(corpus_file, chunker)
        else:
            corpus_chunks = process_chat_logs_whole(corpus_file, chunker)
        
        # Save chunks database for future reference
        chunks_db_file = f"chunk_database{'_whole' if not corpus_solo_content else ''}_{chunk_size}.json"
        with open(chunks_db_file, "w", encoding="utf-8") as f:
            json.dump(corpus_chunks, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved chunks database to {chunks_db_file}")
        
        # Initialize structure for this chunk size
        results["by_chunk_size"][f"size={chunk_size}"] = {
            "num_chunks": len(corpus_chunks),
            "by_k": {}
        }
        
        # Process each question once for max_k
        k_results = {k: [] for k in top_k_values}
        
        for i, question_data in enumerate(questions):
            question = question_data["question"]
            references = question_data["references"]
            
            logger.info(f"Processing question {i+1}/{len(questions)} with chunk_size={chunk_size}: '{question[:50]}...'")
            
            # Retrieve top max_k chunks (only once)
            retrieved_chunks = evaluator.retrieve_top_k_chunks(question, corpus_chunks, k=max_k)
            
            # Store question results for each k value
            if question not in results["by_question"]:
                results["by_question"][question] = {}
            
            if f"chunk_size={chunk_size}" not in results["by_question"][question]:
                results["by_question"][question][f"chunk_size={chunk_size}"] = {}
            
            # For each k value, use the top-k chunks from retrieved_chunks
            for k in top_k_values:
                top_k_chunks = retrieved_chunks[:k]
                
                # Calculate metrics
                precision, recall = evaluator.calculate_precision_recall(top_k_chunks, references)
                f1_score = evaluator.calculate_f1_score(precision, recall)
                
                # Store per-question results
                question_result = {
                    "question_id": i,
                    "question": question,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "retrieved_chunks": [
                        {
                            "content": chunk["content"],
                            "start_index": chunk["start_index"],
                            "end_index": chunk["end_index"],
                            "similarity_score": chunk["similarity_score"]
                        } for chunk in top_k_chunks
                    ]
                }
                k_results[k].append(question_result)
                
                # Store in by_question results
                results["by_question"][question][f"chunk_size={chunk_size}"][f"k={k}"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
        
        # Calculate and store average metrics for each k with this chunk size
        for k in top_k_values:
            precision_values = [r["precision"] for r in k_results[k]]
            recall_values = [r["recall"] for r in k_results[k]]
            f1_values = [r["f1_score"] for r in k_results[k]]
            
            # Calculate average metrics
            avg_precision = sum(precision_values) / len(precision_values)
            avg_recall = sum(recall_values) / len(recall_values)
            avg_f1 = sum(f1_values) / len(f1_values)
            
            # Calculate median metrics
            median_precision = np.median(precision_values)
            median_recall = np.median(recall_values)
            median_f1 = np.median(f1_values)
            
            # Calculate standard deviation metrics
            std_precision = float(np.std(precision_values))
            std_recall = float(np.std(recall_values))
            std_f1 = float(np.std(f1_values))
            
            # Store metrics for this chunk size and k
            results["by_chunk_size"][f"size={chunk_size}"]["by_k"][f"k={k}"] = {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "median_precision": float(median_precision),
                "median_recall": float(median_recall),
                "median_f1": float(median_f1),
                "std_precision": std_precision,
                "std_recall": std_recall,
                "std_f1": std_f1,
                "question_results": k_results[k]
            }
            
            logger.info(f"chunk_size={chunk_size}, k={k} | Avg Precision: {avg_precision:.4f} (±{std_precision:.4f}) | Avg Recall: {avg_recall:.4f} (±{std_recall:.4f}) | Avg F1: {avg_f1:.4f} (±{std_f1:.4f})")
    # Generate summary table of results
    summary = {}
    for chunk_size in chunk_sizes:
        summary[f"size={chunk_size}"] = {}
        for k in top_k_values:
            metrics = results["by_chunk_size"][f"size={chunk_size}"]["by_k"][f"k={k}"]
            summary[f"size={chunk_size}"][f"k={k}"] = {
                "avg_precision": metrics["avg_precision"],
                "avg_recall": metrics["avg_recall"],
                "avg_f1": metrics["avg_f1"],
                "median_precision": metrics["median_precision"],
                "median_recall": metrics["median_recall"],
                "median_f1": metrics["median_f1"]
            }
    
    results["summary"] = summary
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved evaluation results to {output_file}")
    
    return results



if __name__ == "__main__":    
    run_evaluation_pipeline(
        corpus_file="chatlogs.md",
        questions_file="chatlogs_questions.json",
        chunker_type="fixed_token",
        chunk_sizes=[10, 50, 100, 150, 200, 250],
        overlap_percent=0,
        embedding_model="all-MiniLM-L6-v2",
        top_k_values=[1, 3, 5],
        output_file="evaluation_results_wholee.json"
    )