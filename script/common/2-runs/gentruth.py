import configparser # Importing configparser to read configuration files
import os

import numpy as np
import argparse

import pgvector.psycopg
import psycopg

# Logger
import logging

# 
# 
# Distance functions for pgvector
def vector_l2_distance(a, b):
    """
    Calculate the L2 distance (Euclidean distance) between two vectors, without squaring the result.
    """
    return np.linalg.norm(a - b)

def vector_inner_product(a, b):
    """
    Calculate the inner product (dot product) between two vectors.
    """
    return np.dot(a, b)


# 
# Connect to Postgres
def connect_to_postgres(config):
    """
    Connect to PostgreSQL database using the provided configuration.
    """
    db_params = {
        'host': config.get('postgres', 'host'),
        'dbname': config.get('postgres', 'database'),
        'user': config.get('postgres', 'user'),
        'password': config.get('postgres', 'password'),
        "autocommit": True,
        "port": config.getint('postgres', 'port', fallback=5432)
    }
    
    conn = psycopg.connect(**db_params)

    # We need to keep the same connection for this operation
    # Register the pgvector extension
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    pgvector.psycopg.register_vector(conn)

    return conn


# 
# Function to load configuration from a file
def load_configuration(config_path):
    """
    Load the configuration from the given path.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


# 
# Function to check if the index exists in PostgreSQL using pgvector
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate ground truth for pgvector benchmark.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the configuration
    config = load_configuration(args.config)

    # Connect to PostgreSQL
    conn = connect_to_postgres(config)

    # Get the distance operator from the configuration
    distance = config.get('pgvector', 'distance', fallback=None)
    
    if distance is None:
        logging.error("No distance type specified in the configuration.")
        exit(1)
    
    logging.info(f"Running ground truth generation with distance type: {distance}")

    import run
    gt_dicts = run.run_exact_search(conn, config)

    # Save each list

    # Save the ground truth to a file
    # Save the searched ids to the file
    gt_path = config.get('dataset', 'gt_search', fallback=None)
    if gt_path is None:
        logging.error("No ground truth path specified in the configuration.")
        exit(1)

    # Convert the list as the 2D array
    # Check if the ids are 100 length
    if not all(len(x['ids']) == 100 for x in gt_dicts):
        logging.error("Not all ground truth ids have length 100.")
        exit(1)

    gt_ids = np.array([x['ids'] for x in gt_dicts], dtype=np.int32)

    # Make sure it is the right shape
    if gt_ids.ndim != 2:
        logging.error(f"Ground truth data is not 2D, got shape {gt_ids.shape}.")
        exit(1)

    print(f"Saving ground truth to {gt_path} with shape {gt_ids.shape}...")
    np.save(gt_path, gt_ids)

    # Save the scores also, for the regenerate_ground_truth function
    gt_scores_path = gt_path.replace('.npy', '-scores.npy')
    gt_scores = np.array([x['scores'] for x in gt_dicts], dtype=np.float32)

    np.save(gt_scores_path, gt_scores)