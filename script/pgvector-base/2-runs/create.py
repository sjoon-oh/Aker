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
# Function to upload vectors to PostgreSQL using pgvector
def initialize_datastore(config):
    """
    Upload vectors to PostgreSQL using pgvector.
    """

    # Connect to PostgreSQL
    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    logging.info("Connected to PostgreSQL database.")

    # We need to keep the same connection for this operation
    # Register the pgvector extension
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()  # Commit the transaction to ensure the extension is created

    pgvector.psycopg.register_vector(conn)

    conn.execute("DROP TABLE IF EXISTS items CASCADE;")

    # create
    conn.execute(
        f"""
        CREATE TABLE items (
            id SERIAL PRIMARY KEY,
            embedding vector({config.getint('dataset', 'dim')}) NOT NULL
        );"""
    )
    conn.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN;")

    # Close connection
    conn.commit()
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

    # Initialize the datastore
    try:
        initialize_datastore(config)
        logging.info("Datastore initialized successfully.")

    except Exception as e:
        logging.error(f"Failed to initialize datastore: {e}")
        exit(1)

    # 
    # Check the table
    conn = connect_to_postgres(config)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items;")
    rows = cursor.fetchall()
    logging.info(f"Table 'items' created successfully with {len(rows)} rows.")