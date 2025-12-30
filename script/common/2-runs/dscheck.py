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
def check_index_exists(conn, config):
    """
    Check if the index exists in PostgreSQL using pgvector.
    """
    
    cursor = conn.cursor()

    index_name = config.get('pgvector', 'index_name', fallback='')
    if index_name == '':
        index_type = config.get('pgvector', 'type', fallback=None)
        if index_type is None:
            logging.error("No index type specified in the configuration.")
            exit(1)
        index_name = f"index_${index_type}_idx"

    # Check if the index exists as the index_name
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = %s;
    """, (index_name,))
    index_lists = cursor.fetchall()

    # Make it a flat list
    index_lists = [item[0] for item in index_lists]
    logging.info(f"Checking for index '{index_name}' in the database: {index_lists}")

    is_ready = False

    if index_name in index_lists:
        logging.info(f"Index '{index_name}' exists in the database.")
        is_ready = True
    else:
        logging.error(f"Index '{index_name}' does not exist in the database.")

    return is_ready


# 
# 
if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Check if the index exists in PostgreSQL.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = load_configuration(args.config)

    # Connect to PostgreSQL
    conn = connect_to_postgres(config)
    if not conn:
        logging.error("Failed to connect to PostgreSQL database.")
        exit(1)
    
    # Check if the index exists
    try:
        check_index_exists(conn, config)
        logging.info("Index check completed successfully.")

    except Exception as e:
        logging.error(f"Failed to check index: {e}")
        exit(1)

    # Close the connection
    conn.close()
