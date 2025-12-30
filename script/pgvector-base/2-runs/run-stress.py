import configparser # Importing configparser to read configuration files
import os

import numpy as np
import pickle
import argparse

import random

import pgvector.psycopg
import psycopg

# Logger
import logging

import dscheck

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
def get_distance_operator(config):
    """
    Get the distance operator from the configuration.
    """
    distance = config.get('pgvector', 'distance', fallback=None)

    if distance is None:
        logging.error("No distance type specified in the configuration.")
        raise ValueError("Distance type must be specified in the configuration under 'pgvector' section.")
    
    if distance == 'vector_l2_ops':
        return '<->'
    elif distance == 'vector_cosine_ops':
        return '<=>'
    elif distance == 'vector_ip_ops':
        return '<#>'
    else:
        logging.error(f"Unsupported distance type: {distance}. Supported types are 'vector_l2_ops' and 'vector_cosine_ops'.")
        raise ValueError(f"Unsupported distance type: {distance}. Supported types are 'vector_l2_ops' and 'vector_cosine_ops'.")
    


# 
# 
# Run the main script
if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Check if the index exists in PostgreSQL.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--invalidate', type=float, default=0.0, help='Percentage of cache to invalidate (0.0 to 1.0). Default is 0.0 (no invalidation).')
    parser.add_argument('--entries', type=int, default=1000, help='Number of entries estimated. Default is 1000.')

    args = parser.parse_args()
    entries_estimated = args.entries

    # Load configuration
    config = load_configuration(args.config)

    # Connect to PostgreSQL
    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    is_ready = dscheck.check_index_exists(conn, config)
    if is_ready:
        logging.info("Index check completed successfully.")
    else:
        logging.error("Index check failed. Please ensure the index exists in the database.")

    #
    # Check if the dataset trace exists.
    trace_path = config.get('dataset', 'gt_trace', fallback='trace.pkl')
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file {trace_path} does not exist. Please ensure the trace file is present at the specified path.")
    
    workload_type = config.get('workload', 'wtype', fallback=None)
    if workload_type is None:
        raise ValueError("Workload type must be specified in the configuration under 'dataset' section.")
    
    # The workload_type must be one of the 'stress-delete', 'stress-insert', 'stress-mixed'
    if workload_type not in ['stress-delete', 'stress-insert', 'stress-mixed']:
        raise ValueError(f"Unsupported workload type: {workload_type}. Supported types are 'stress-delete', 'stress-insert', and 'stress-mixed'.")

    # 
    # Check the number of the saved rows
    cursor.execute("SELECT COUNT(*) FROM items;")
    row_count = cursor.fetchone()[0]
    logging.info(
        f"Number of rows in the 'items' table: {row_count}. This should match the number of rows in the dataset."
    )

    max_vector_id = 0
    cursor.execute("SELECT MAX(id) FROM items;")
    max_vector_id = cursor.fetchone()[0]

    # 
    # Prepare the sql string
    limit = config.getint('workload', 'limit', fallback=10)
    distance_operator = get_distance_operator(config)

    search_sql = "SELECT id, embedding " + distance_operator + " %s AS _score FROM items ORDER BY _score LIMIT %s;"
    insert_sql = "INSERT INTO items (id, embedding) VALUES (%s, %s);"

    # 
    # Index type?
    index_type = config.get('pgvector', 'type', fallback=None)
    
    # Assure the index type is supported
    if index_type not in ['ivfflat', 'hnsw']:
        raise ValueError(f"Unsupported index type: {index_type}. Supported types are 'ivfflat' and 'hnsw'.")

    if index_type == 'hnsw':
        # For HNSW, we need to set the ef_construction parameter
        ef_search = config.getint('pgvector', 'ef_search', fallback=None)
        cursor.execute(f"SET hnsw.ef_search = {ef_search};")

    elif index_type == 'ivfflat':
        # For IVFFlat, we need to set the nprobe parameter
        nprobe = config.getint('pgvector', 'nprobe', fallback=None)
        cursor.execute(f"SET ivfflat.probes = {nprobe};")

    logging.info(f"Starting workload type: {workload_type} with index type: {index_type}.")

    # Load pickle trace
    trace = None
    search_requests = []
    insert_requests = []

    with open(trace_path, 'rb') as f:
        trace = pickle.load(f)
    
    # We do not check the ground truth for the stress tests.
    # Instead, we just run as many times as possible.

    search_trace = trace['search']
    insert_trace = trace['insert']

    insert_trace = insert_trace[:len(search_trace)]

    logging.info(f"Loaded {len(search_trace)} search requests and {len(insert_trace)} insert requests from the trace.")

    # Randomly shuffle the requests
    # random.shuffle(search_trace)
    # random.shuffle(insert_trace)

    # Do the prefill of the first 10% of the insert requests
    prefill_requests = search_trace

    import copy
    remaining_search_requests = copy.deepcopy(search_trace)
    
    # random.shuffle(remaining_search_requests)
    logging.info(f"Prefilling the database with {len(search_trace)} insert requests.")

    import time
    show_progress_percentage = 1
    for i, request in enumerate(prefill_requests):
        search_vector = request['payload']
        
        # Ensure search_vector is a numpy array
        if not isinstance(search_vector, np.ndarray):
            raise TypeError(f"Expected search_vector to be a numpy array, got {type(search_vector)} instead.")

        # Execute the search query
        cursor.execute(search_sql, (search_vector, limit), 
                        binary=True, prepare=True)
        results = cursor.fetchall()

        percentage = (i + 1) / len(prefill_requests) * 100
        if percentage >= show_progress_percentage:
            logging.info(f"Prefill Progress: {percentage:.2f}% completed.")
            show_progress_percentage += 1

    avg_recall = 0.0

    # Run the next phase of the workload
    search_result = []
    if workload_type == 'stress-delete':

        invalidate_percentage = args.invalidate
        
        # 
        # Make the topkache reset
        cursor.execute(f"SELECT topkache_invalidate_random({invalidate_percentage});")
        logging.info(f">> TopKache reset after prefill: {invalidate_percentage*100:.2f}% invalidated.")

        # Next, we run the search requests
        show_progress_percentage = 1
        workload_start = time.perf_counter()
        for i, request in enumerate(remaining_search_requests):
            search_vector = request['payload']
            
            # Ensure search_vector is a numpy array
            if not isinstance(search_vector, np.ndarray):
                raise TypeError(f"Expected search_vector to be a numpy array, got {type(search_vector)} instead.")

            request_start = time.perf_counter()
            # Execute the search query
            cursor.execute(search_sql, (search_vector, limit), 
                            binary=True, prepare=True)
            results = cursor.fetchall()
            request_end = time.perf_counter()

            # Process results
            result_ids = [row[0] for row in results]
            result_scores = [row[1] for row in results]

            # Process results
            search_result.append({
                'operation': 'search',
                'latency': request_end - request_start,
                'latency_accumulated': request_end - workload_start,
                'qps_moment': (i + 1) / (request_end - workload_start) if (request_end - workload_start) > 0 else 0,
                'gt_ids': request['gt_ids'],
                'gt_scores': request['gt_scores'],
                'result_ids': result_ids,
                'result_scores': result_scores
            })

            percentage = (i + 1) / len(remaining_search_requests) * 100

            # If the workload exceeds each 10%, print the progress
            if percentage >= show_progress_percentage:
                logging.info(f"Progress: {percentage:.2f}% completed.")
                show_progress_percentage += 1

                elapsed = time.perf_counter() - workload_start
                estimated_remaining = (elapsed / percentage) * (100 - percentage) if percentage > 0 else 0
                logging.info(f"Estimated remaining time: {estimated_remaining:.2f} seconds. (QPS: {(i + 1) / elapsed if elapsed > 0 else 0:.2f})")

        # 
        # Recall
        recalls = []
        for result in search_result:
            if result['operation'] != 'search':
                continue
            else:
                gt_ids = set(result['gt_ids'][:limit])
                result_ids = set(result['result_ids'])
                recalls.append(len(gt_ids.intersection(result_ids)) / limit)
        
        avg_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
        logging.info(f"Average recall@{limit}: {avg_recall:.4f}")

    elif workload_type == 'stress-insert':
        # First, insert the insert vectors enough to fill the log

        show_progress_percentage = 1
        sub_insert_trace = insert_trace[:len(search_trace)]
        for i, request in enumerate(sub_insert_trace):

            insert_vector = request['payload']
            max_vector_id += 1
            cursor.execute(insert_sql, (max_vector_id, insert_vector), 
                            binary=True, prepare=True)
            
            percentage = (i + 1) / len(sub_insert_trace) * 100
            if percentage >= show_progress_percentage:
                logging.info(f"Insert Progress: {percentage:.2f}% completed.")
                show_progress_percentage += 1

        logging.info(f"Inserted {len(sub_insert_trace)} vectors to fill the log.")

        # 
        # Make the topkache reset
        invalidate_percentage = args.invalidate
        
        cursor.execute(f"SELECT topkache_invalidate_random({invalidate_percentage});")
        logging.info(f">> TopKache reset after prefill: {invalidate_percentage*100:.2f}% invalidated.")

        # Next, we run the search requests
        show_progress_percentage = 1
        workload_start = time.perf_counter()
        for i, request in enumerate(remaining_search_requests):
            search_vector = request['payload']
            
            # Ensure search_vector is a numpy array
            if not isinstance(search_vector, np.ndarray):
                raise TypeError(f"Expected search_vector to be a numpy array, got {type(search_vector)} instead.")

            request_start = time.perf_counter()
            # Execute the search query
            cursor.execute(search_sql, (search_vector, limit), 
                            binary=True, prepare=True)
            results = cursor.fetchall()
            request_end = time.perf_counter()

            # Process results
            result_ids = [row[0] for row in results]
            result_scores = [row[1] for row in results]

            # Process results
            search_result.append({
                'operation': 'search',
                'latency': request_end - request_start,
                'latency_accumulated': request_end - workload_start,
                'qps_moment': (i + 1) / (request_end - workload_start) if (request_end - workload_start) > 0 else 0,
                'gt_ids': request['gt_ids'],
                'gt_scores': request['gt_scores'],
                'result_ids': result_ids,
                'result_scores': result_scores
            })

            percentage = (i + 1) / len(remaining_search_requests) * 100

            # If the workload exceeds each 10%, print the progress
            if percentage >= show_progress_percentage:
                logging.info(f"Progress: {percentage:.2f}% completed.")
                show_progress_percentage += 1

                elapsed = time.perf_counter() - workload_start
                estimated_remaining = (elapsed / percentage) * (100 - percentage) if percentage > 0 else 0
                logging.info(f"Estimated remaining time: {estimated_remaining:.2f} seconds. (QPS: {(i + 1) / elapsed if elapsed > 0 else 0:.2f})")

        # Recall
        # Running the recall evaluation
        recalls = []
        for result in search_result:
            if result['operation'] != 'search':
                continue
            else:
                gt_ids = set(result['gt_ids'][:limit])
                result_ids = set(result['result_ids'])
                recalls.append(len(gt_ids.intersection(result_ids)) / limit)

        # Average recall
        avg_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
        logging.info(f"Average recall@{limit}: {avg_recall:.4f}")
        # Add the average recall to each search result

    elif workload_type == 'stress-mixed':

        # 
        # Make the topkache reset
        invalidate_percentage = args.invalidate

        # First, insert the insert vectors enough to fill the log
        show_progress_percentage = 1
        sub_insert_trace = insert_trace[:len(search_trace)]
        for i, request in enumerate(sub_insert_trace):

            insert_vector = request['payload']
            max_vector_id += 1
            cursor.execute(insert_sql, (max_vector_id, insert_vector), 
                            binary=True, prepare=True)
            
            percentage = (i + 1) / len(sub_insert_trace) * 100
            if percentage >= show_progress_percentage:
                logging.info(f"Insert Progress: {percentage:.2f}% completed.")
                show_progress_percentage += 1

        logging.info(f"Inserted {len(sub_insert_trace)} vectors to fill the log.")

        cursor.execute(f"SELECT topkache_invalidate_random({invalidate_percentage});")
        logging.info(f">> TopKache reset after prefill: {invalidate_percentage*100:.2f}% invalidated.")

        # Next, we run the search requests
        workload_start = time.perf_counter()
        for i, request in enumerate(remaining_search_requests):
            search_vector = request['payload']
            
            # Ensure search_vector is a numpy array
            if not isinstance(search_vector, np.ndarray):
                raise TypeError(f"Expected search_vector to be a numpy array, got {type(search_vector)} instead.")

            request_start = time.perf_counter()
            # Execute the search query
            cursor.execute(search_sql, (search_vector, limit), 
                            binary=True, prepare=True)
            results = cursor.fetchall()
            request_end = time.perf_counter()

            # Process results
            search_result.append({
                'operation': 'search',
                'latency': request_end - request_start,
                'latency_accumulated': request_end - workload_start,
                'qps_moment': (i + 1) / (request_end - workload_start) if (request_end - workload_start) > 0 else 0
            })

            percentage = (i + 1) / len(remaining_search_requests) * 100

            # If the workload exceeds each 10%, print the progress
            if percentage >= show_progress_percentage:
                logging.info(f"Progress: {percentage:.2f}% completed.")
                show_progress_percentage += 1

                elapsed = time.perf_counter() - workload_start
                estimated_remaining = (elapsed / percentage) * (100 - percentage) if percentage > 0 else 0
                logging.info(f"Estimated remaining time: {estimated_remaining:.2f} seconds. (QPS: {(i + 1) / elapsed if elapsed > 0 else 0:.2f})")


    total_average_latency = np.mean([result['latency'] for result in search_result])

    search_latencies = [result['latency'] for result in search_result if result['operation'] == 'search']
    if not search_latencies:
        logging.warning("No search latencies found. Ensure that the trace contains search operations.")
        average_search_latency = 0.0
        p50_search_latency = 0.0
        p99_search_latency = 0.0
    else:
        average_search_latency = np.mean(search_latencies)
        p50_search_latency = np.percentile(search_latencies, 50)
        p99_search_latency = np.percentile(search_latencies, 99)

    logging.info(f"Average search latency: {average_search_latency:.4f} seconds, P50: {p50_search_latency:.4f}, P99: {p99_search_latency:.4f}")


    # Write to a file
    report_file = "report.csv"

    # Headers: 
    # Name, workload type, search params, qps, avg latency, 50%ile latency, 99%ile latency, avg recall
    with open(report_file, 'w') as f:
        # f.write("Name,Workload Type,Search Params,QPS,Avg Latency (s),50%ile Latency (s),99%ile Latency (s),Avg Recall\n")
        f.write("Name\tWorkload Type\tSearch Params\tQPS\t")
        f.write(
            "Avg Search Latency (s)\t50%ile Search Latency (s)\t99%ile Search Latency (s)\t"
        )
        f.write(
            "Avg Delete Latency (s)\t50%ile Delete Latency (s)\t99%ile Delete Latency (s)\t"
        )
        f.write(
            "Avg Insert Latency (s)\t50%ile Insert Latency (s)\t99%ile Insert Latency (s)\t"
        )
        f.write("Avg Recall\n")

        workload_name = config.get('workload', 'name', fallback='unknown')
        search_params = ""
        if index_type == 'ivfflat':
            search_params = f"nprobe={config.getint('pgvector', 'nprobe', fallback=None)}"

        elif index_type == 'hnsw':
            search_params = f"ef_search={config.getint('pgvector', 'ef_search', fallback=None)}"

        else:
            search_params = "unknown"
        
        qps = len(search_result) / sum([result['latency'] for result in search_result]) if sum([result['latency'] for result in search_result]) > 0 else 0
        # The qps unit is in queries per second

        f.write(f"{workload_name}\t{workload_type}\t{search_params}\t{qps:.2f}\t")
        f.write(f"{average_search_latency:.4f}\t{p50_search_latency:.4f}\t{p99_search_latency:.4f}\t{avg_recall:.4f}\t")

    # Export the search results to a file
    search_results_file = "search-results.pkl"
    with open(search_results_file, 'wb') as f:
        pickle.dump(search_result, f)

    logging.info(f"Search results written to {search_results_file}.")

    # 
    # Export extra traces
    recall_trace_file = "trace-extract-info.csv"
    with open(recall_trace_file, 'w') as f:
        for i, result in enumerate(search_result):
            operation_type = result['operation']

            # Only remain the first letter
            operation_type = operation_type[0].upper() if operation_type else 'U'
            
            latency = result['latency']
            qps_moment = result['qps_moment']