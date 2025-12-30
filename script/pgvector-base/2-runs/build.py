import configparser # Importing configparser to read configuration files
import os

import numpy as np
import argparse

import pgvector.psycopg
import psycopg

# Logger
import logging

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
# 
def build_index(config):
    """
    Build the index in PostgreSQL using pgvector.
    """

    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    # Base path
    base_path = config.get('dataset', 'base', fallback='.')

    # 
    # The base path can contain asterisks, so we need to handle that
    if '*' in base_path:
        # If the base path contains asterisks, we need to find all files matching the pattern
        import glob
        matching_files = sorted(glob.glob(base_path))
        
        if not matching_files:
            logging.error(f"No files found matching pattern: {base_path}")
            exit(1)
        
        # We will process each file separately
        for file_path in matching_files:
            logging.info(f" >> Existing file: {file_path}")
            # Here you would add the logic to process each file
            # For example, you could read the file and upload its contents to the database
    else:
        # If the base path does not contain asterisks, we can use it directly
        logging.info(f" >> Existing file: {base_path}")
        # Here you would add the logic to process the file
        # For example, you could read the file and upload its contents to the database

    # 
    # We only do the two index types for pgvector
    index_type = config.get('pgvector', 'type', fallback=None)
    index_name = config.get('pgvector', 'index_name', fallback="")
    distance_type = config.get('pgvector', 'distance', fallback=None)

    # The distance should be the one supported by pgvector
    if distance_type not in ['vector_l2_ops', 'vector_ip_ops', 'vector_cosine_ops', 'vector_l1_ops']:
        logging.error(f"Unsupported distance type: {distance_type}. Supported types are 'vector_l2_ops' and 'vector_cosine_ops'.")
        exit(1)

    if index_type not in ['hnsw', 'ivfflat']:
        logging.error(f"Unsupported index type: {index_type}. Supported types are 'hnsw' and 'ivfflat'.")
        exit(1)

    if index_name == "":
        index_name = f"items_{index_type}_idx"

    conn.execute("SET max_parallel_workers = 96;")
    conn.execute("SET max_parallel_maintenance_workers = 96;")

    if index_type == 'hnsw':
        m = config.getint('pgvector', 'm', fallback=16)
        ef_construct = config.getint('pgvector', 'ef_construction', fallback=64)
        cursor.execute(
            f"""
            CREATE INDEX CONCURRENTLY {index_name} ON items USING hnsw (embedding {distance_type})
            WITH (m = {m}, ef_construction = {ef_construct});
            """
        )
    elif index_type == 'ivfflat':
        nlist = config.getint('pgvector', 'nlist', fallback=100)
        cursor.execute(
            f"""
            CREATE INDEX CONCURRENTLY {index_name} ON items USING ivfflat (embedding {distance_type})
            WITH (lists = {nlist});
            """
        )

    print(f"Index {index_name} created successfully with type {index_type} and distance {distance_type}.")

    cursor.close()
    conn.close()


# 
# Main function to run the upload process
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Add argument parsing
    parser = argparse.ArgumentParser(description="Upload vectors to PostgreSQL with pgvector.")
    parser.add_argument('--config', type=str, help='Path to the configuration file.')

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        logging.error(f"Configuration file {config_path} does not exist.")
        exit(1)

    config = load_configuration(config_path)
    logging.info(f"Loaded configuration from {config_path}")

    # Build the index
    try:
        build_index(config)
        logging.info("Index built successfully.")
    except Exception as e:
        logging.error(f"Failed to build index: {e}")
        exit(1)
    
    logging.info("Indexing completed successfully.")