import configparser # Importing configparser to read configuration files
import os

import numpy as np
import pickle
import argparse

import random

import pgvector.psycopg
import psycopg

# Logger  k
import logging

#
# 
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


# Function to load configuration from a file
def load_configuration(config_path):
    """
    Load the configuration from the given path.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


# 
# 
# run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth for pgvector benchmark.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    config = load_configuration(args.config)

    # 
    # This script fixes the "holes", which if the loaded trace file contains the 
    # search operations (that went through the workload generation script) but does not
    # have the full gt_ids, it will attempt to reconstruct them.
    # 
    # While looping through the trace file, it will check if the gt_ids are present.
    # If not, it will attempt to reconstruct them by querying the database.
    # 
    # 
    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    # Load the trace file
    trace_file_path = config.get('dataset', 'gt_trace')
    with open(trace_file_path, 'rb') as f:
        trace_data = pickle.load(f)

    if trace_file_path is None:
        logging.error("Trace file path is not specified in the configuration.")
        exit(1)

    logging.info(f"Loaded trace data from {trace_file_path}.")
    logging.info(f"Trying to fix the trace data: {trace_file_path}.")

    # 
    # Iterate through the trace data and fix the gt_ids
    fixed = 0
    for i, entry in enumerate(trace_data):

        # First, check the operation type is the "search"
        operation_type = 'search'
        if entry['operation'] != operation_type:
            continue

        # Check if the entry is zero length
        if (entry['gt_ids'] is None) or (len(entry['gt_ids']) == 0):
            # If gt_ids are missing, query the database to reconstruct them
            query = "SELECT id FROM vectors WHERE vector = %s;"
            vector = entry['payload']

            try:
                query_string = f"SELECT id, embedding <-> \'{vector.tolist()}\' AS _score FROM items ORDER BY _score LIMIT 100;"

                with conn.transaction():
                    cursor.execute("SET LOCAL enable_indexscan = off")
                    cursor.execute(query_string)

                searched = cursor.fetchall()

                # 
                ids = np.array([x[0] for x in searched], dtype=np.int32)
                scores = np.array([x[1] for x in searched], dtype=np.float32)

                entry['gt_ids'] = ids
                entry['gt_scores'] = scores

                fixed += 1

                logging.info(f"Fixed entry {i}th.")

            except Exception as e:
                logging.error(f"Error while querying the database: {e}")
                continue
    
    logging.info(f"Total entries in the workload file: {len(trace_data)}, Total fixed entries: {fixed}")
    logging.info(f"Successfully fixed {fixed} items in the trace data and saved it back to the file.") 

    # Recheck if the trace data is fixed
    for entry in trace_data:

        operation_type = 'search'
        if entry['operation'] != operation_type:
            continue

        # Check if the entry is zero length
        if (entry['gt_ids'] is None) or (len(entry['gt_ids']) == 0):
            logging.error(f"Some gt_ids are still missing after the fix: \n{entry['gt_ids']}")
            raise ValueError("Some gt_ids are still missing after the fix.")

    # Save the fixed trace data back to the file
    # The file should be overwritten with the fixed data
    with open(trace_file_path, 'wb') as f:
        pickle.dump(trace_data, f)

    logging.info(f"Successfully fixed {fixed} items in the trace data and saved it back to the file.")
    cursor.close()
    conn.close()
