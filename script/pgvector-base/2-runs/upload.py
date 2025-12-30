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

def copy_base_2(config):
    """
    Upload vectors to PostgreSQL using pgvector.
    Now supports asterisk patterns in file paths with memory-efficient processing.
    """

    def process_and_insert_vectors(vectors, start_id, split_num, workload_type, cursor):
        """
        Helper function to process vectors and insert them into database.
        Returns (records_inserted, next_id, should_stop)
        """
        records = []
        current_id = start_id
        
        for vector in vectors:
            # Check if we need to stop due to split_num (for workloadb)
            if workload_type == 'workloadb' and split_num is not None and current_id >= split_num:
                break
            
            # Ensure vector is a list or tuple
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
                records.append((current_id, np.array(vector)))
            elif not isinstance(vector, (list, tuple)):
                logging.error(f"Vector at index {current_id} is not a list or tuple: {vector}")
                exit(1)
            
            current_id += 1
        
        # Insert records if any
        records_inserted = 0
        if records:
            with cursor.copy(
                "COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["integer", "vector"])
                for record in records:
                    copy.write_row(record)

                if records_inserted == 100:
                    logging.info(f"Inserted {records_inserted / (len(records)) * 100:.2f}% records into 'items' table.")
                
                records_inserted = records_inserted + 1
        
        # Check if we should stop processing more files
        should_stop = (workload_type == 'workloadb' and split_num is not None and current_id >= split_num)
        
        return records_inserted, current_id, should_stop

    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    # Base path
    base_path = config.get('dataset', 'base', fallback='.')
    split_num = config.getint('dataset', 'split_num', fallback=None)
    workload_type = config.get('workload', 'wtype', fallback=None)

    # You should have your workload type set
    if workload_type is None:
        logging.error("Workload type is not set in the configuration. Please specify a workload type.")
        exit(1)

    # Validate workloadb configuration
    if workload_type == 'workloadb' and split_num is None:
        logging.error("Split number is not set in the configuration. Please specify a split number.")
        exit(1)

    total_inserted = 0
    current_id = 0

    # Check if base_path contains asterisk (pattern)
    if '*' in base_path:
        import glob
        import gc
        from pathlib import Path
        
        # Find all files matching the pattern
        matching_files = sorted(glob.glob(base_path))

        print(f"Matching files: {matching_files}")
        
        if not matching_files:
            logging.error(f"No files found matching pattern: {base_path}")
            exit(1)
        
        logging.info(f"Found {len(matching_files)} files matching pattern: {base_path}")
        for f in matching_files:
            logging.info(f"  - {Path(f).name}")
        
        # Process files one by one with memory-efficient approach
        for file_idx, file_path in enumerate(matching_files):
            logging.info(f"Processing file {file_idx + 1}/{len(matching_files)}: {Path(file_path).name}")
            
            try:
                # Load single file
                vectors = np.load(file_path)
                logging.info(f"Loaded {len(vectors)} vectors from {Path(file_path).name}")
                
                # Process and insert vectors from this file
                records_inserted, current_id, should_stop = process_and_insert_vectors(
                    vectors, current_id, split_num, workload_type, cursor
                )
                
                total_inserted += records_inserted
                if records_inserted > 0:
                    logging.info(f"Inserted {records_inserted} records from {Path(file_path).name}. Total so far: {total_inserted}")
                
                # Clean up memory immediately
                del vectors
                gc.collect()
                
                # Stop if we've reached split_num for workloadb
                if should_stop:
                    logging.info(f"Reached split_num limit ({split_num}). Stopping file processing.")
                    break
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                exit(1)
        
        logging.info(f"Total vectors inserted from all files: {total_inserted}")
        
    else:
        # Single file processing
        if not os.path.exists(base_path):
            logging.error(f"Base path {base_path} does not exist.")
            exit(1)
        
        # Load the npy file
        vectors = np.load(base_path)
        logging.info(f"Loaded {len(vectors)} vectors from {base_path}")

        # Process and insert all vectors from single file
        records_inserted, _, _ = process_and_insert_vectors(
            vectors, current_id, split_num, workload_type, cursor
        )
        
        total_inserted = records_inserted
        logging.info(f"Inserted {records_inserted} records from single file.")

    # Log workload type results
    if workload_type == 'workloada':
        logging.info("Workload type is 'workloada'. All vectors uploaded.")
    elif workload_type == 'workloadb':
        logging.info(f"Workload type is 'workloadb'. Used first {total_inserted} vectors.")

    # Count the number of rows in the table
    cursor.execute("SELECT COUNT(*) FROM items;")
    row_count = cursor.fetchone()[0]
    logging.info(f"Number of rows in 'items' table: {row_count}")

# 
# Function to upload vectors to PostgreSQL using pgvector
def copy_base(config):
    """
    Upload vectors to PostgreSQL using pgvector.
    """

    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    # Base path
    base_path = config.get('dataset', 'base', fallback='.')
    split_num = config.getint('dataset', 'split_num', fallback=None)

    workload_type = config.get('workload', 'wtype', fallback=None)

    # You should have your workload type set
    if workload_type is None:
        logging.error("Workload type is not set in the configuration. Please specify a workload type.")
        exit(1)

    if not os.path.exists(base_path):
        logging.error(f"Base path {base_path} does not exist.")
        exit(1)
    
    else:
        # Load the npy file
        vectors = np.load(config.get('dataset', 'base'))
        logging.info(f"Loaded {len(vectors)} vectors from {config.get('dataset', 'base')}")

        records = []
        for idx, vector in enumerate(vectors):

            # Ensure vector is a list or tuple
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
                records.append((idx, np.array(vector)))

            elif not isinstance(vector, (list, tuple)):
                logging.error(f"Vector at index {idx} is not a list or tuple: {vector}")
                exit(1)

        # If split configuration is set, just use the first split_num vectors
        if workload_type == 'workloada':
            logging.info("Workload type is 'workloada'. Using all vectors for upload.")
        elif workload_type == 'workloadb':
            logging.info("Workload type is 'workloadb'. Using split vectors for upload.")
            if split_num is not None:
                if split_num > len(records):
                    logging.error(f"Split number {split_num} exceeds the number of records {len(records)}.")
                    exit(1)
            else:
                logging.error("Split number is not set in the configuration. Please specify a split number.")
                exit(1)

            records = records[:split_num]
            logging.info(f"Using first {split_num} vectors for upload.")

        # 
        # 
        logging.info(f"Inserting {len(records)} records into 'items' table.")
        with cursor.copy(
            "COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["integer", "vector"])
            for record in records:
                copy.write_row(record)
        
        # 

    # 
    # Count the number of rows in the table
    cursor.execute("SELECT COUNT(*) FROM items;")
    row_count = cursor.fetchone()[0]
    logging.info(f"Number of rows in 'items' table: {row_count}")


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

    copy_base_2(config)
    logging.info("Base vectors uploaded successfully.")

    # Check the table items
    conn = connect_to_postgres(config)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM items;")

    row_count = cursor.fetchone()[0]
    logging.info(f"Table 'items' inserted successfully with {row_count} rows.")