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
    args = parser.parse_args()

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
    delete_sql = "DELETE FROM items WHERE id = %s;"

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
    with open(trace_path, 'rb') as f:
        trace = pickle.load(f)

    # Before running, check if the search trace contains all the ground truth ids
    # The GT should be in the list (elements should be in the list)
    # Check for only the 'search' operation
    for request in trace:
        operation = request['operation']
        if operation == 'search':
            gt_ids = request.get('gt_ids', None)
            if gt_ids is None or len(gt_ids) == 0:
                logging.error("Ground truth IDs are missing in the search request. Please ensure the trace contains ground truth IDs.")
                raise ValueError(f"Ground truth IDs are missing in the search request.")
    
    logging.info(f"Loaded trace with {len(trace)} requests.")

    import time

    search_result = []
    workload_start = time.perf_counter()

    show_progress_percentage = 1
    for i, request in enumerate(trace):
        operation = request['operation']

        # print(f"Executing request: {i}")

        request_start = time.perf_counter()
        if operation == 'search':
            search_vector = request['payload']

            # Ensure search_vector is a numpy array
            if not isinstance(search_vector, np.ndarray):
                raise TypeError(f"Expected search_vector to be a numpy array, got {type(search_vector)} instead.")

            # Execute the search query
            cursor.execute(search_sql, (search_vector, limit), 
                           binary=True, prepare=True)
            results = cursor.fetchall()
            request_end = time.perf_counter()

            # Process results
            result_ids = [row[0] for row in results]
            result_scores = [row[1] for row in results]

            # Store the result
            search_result.append({
                'operation': operation,
                'search_vector': search_vector,
                'result_ids': result_ids,
                'result_scores': result_scores,
                'gt_ids': request['gt_ids'],
                'gt_scores': request['gt_scores'],
                'latency': request_end - request_start,
                'latency_accumulated': request_end - workload_start,
                'qps_moment': (i + 1) / (request_end - workload_start) if (request_end - workload_start) > 0 else 0
            })

        elif operation == 'insert':
            vector = request['payload']
            id = max_vector_id + 1

            cursor.execute(insert_sql, (id, vector), 
                           binary=True, prepare=True)
            request_end = time.perf_counter()
            search_result.append({
                'operation': operation,
                'search_vector': None,
                'result_ids': [],
                'result_scores': [],
                'gt_ids': [],
                'gt_scores': [],
                'latency': request_end - request_start,
                'latency_accumulated': request_end - workload_start,
                'qps_moment': (i + 1) / (request_end - workload_start) if (request_end - workload_start) > 0 else 0
            })
            max_vector_id += 1
            request_end = time.perf_counter()
        
        elif operation == 'delete':
            id = request['payload']
            cursor.execute(delete_sql, (id,), 
                           binary=True, prepare=True)
            request_end = time.perf_counter()
            search_result.append({
                'operation': operation,
                'search_vector': None,
                'result_ids': [],
                'result_scores': [],
                'gt_ids': [],
                'gt_scores': [],
                'latency': request_end - request_start,
                'latency_accumulated': request_end - workload_start,
                'qps_moment': (i + 1) / (request_end - workload_start) if (request_end - workload_start) > 0 else 0
            })
        else:
            logging.error(f"Unsupported operation: {operation}. Supported operations are 'search', 'insert', and 'delete'.")
            raise ValueError(f"Unsupported operation: {operation}. Supported operations are 'search', 'insert', and 'delete'.")

        percentage = (i + 1) / len(trace) * 100

        # If the workload exceeds each 10%, print the progress
        if percentage >= show_progress_percentage:
            logging.info(f"Progress: {percentage:.2f}% completed.")
            show_progress_percentage += 1

    workload_end = time.perf_counter()
    total_latency = workload_end - workload_start

    qps = len(trace) / total_latency if total_latency > 0 else 0
        
    logging.info(f"Workload type: {workload_type} completed. Total requests: {len(trace)}.")
    logging.info(f"Total latency: {total_latency:.4f} seconds, QPS: {qps:.2f}")

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

    delete_latencies = [result['latency'] for result in search_result if result['operation'] == 'delete']
    if not delete_latencies:
        logging.warning("No delete latencies found. Ensure that the trace contains delete operations.")
        average_delete_latency = 0.0
        p50_delete_latency = 0.0
        p99_delete_latency = 0.0
    else:
        average_delete_latency = np.mean(delete_latencies)
        p50_delete_latency = np.percentile(delete_latencies, 50)
        p99_delete_latency = np.percentile(delete_latencies, 99)

    insert_latencies = [result['latency'] for result in search_result if result['operation'] == 'insert']
    if not insert_latencies:
        logging.warning("No insert latencies found. Ensure that the trace contains insert operations.")
        average_insert_latency = 0.0
        p50_insert_latency = 0.0
        p99_insert_latency = 0.0
    else:
        average_insert_latency = np.mean(insert_latencies)
        p50_insert_latency = np.percentile(insert_latencies, 50)
        p99_insert_latency = np.percentile(insert_latencies, 99)

    logging.info(f"Average search latency: {average_search_latency:.4f} seconds, P50: {p50_search_latency:.4f}, P99: {p99_search_latency:.4f}")
    logging.info(f"Average delete latency: {average_delete_latency:.4f} seconds, P50: {p50_delete_latency:.4f}, P99: {p99_delete_latency:.4f}")
    logging.info(f"Average insert latency: {average_insert_latency:.4f} seconds, P50: {p50_insert_latency:.4f}, P99: {p99_insert_latency:.4f}")

    # Running the recall evaluation
    recalls = []
    for result in search_result:
        if result['operation'] != 'search':
            continue
        else:
            gt_ids = set(result['gt_ids'][:limit])
            result_ids = set(result['result_ids'])
            recalls.append(len(gt_ids.intersection(result_ids)) / limit)

    avg_recall = np.mean(recalls) if recalls else 0.0
    logging.info(f"Average recall: {avg_recall:.4f}, total records: {len(recalls)}")

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
        
        f.write(f"{workload_name}\t{workload_type}\t{search_params}\t{qps:.2f}\t")
        f.write(f"{average_search_latency:.4f}\t{p50_search_latency:.4f}\t{p99_search_latency:.4f}\t")
        f.write(f"{average_delete_latency:.4f}\t{p50_delete_latency:.4f}\t{p99_delete_latency:.4f}\t")
        f.write(f"{average_insert_latency:.4f}\t{p50_insert_latency:.4f}\t{p99_insert_latency:.4f}\t")
        f.write(f"{avg_recall:.4f}\n")

    # Export the search results to a file
    search_results_file = "search-results.pkl"
    with open(search_results_file, 'wb') as f:
        pickle.dump(search_result, f)

    logging.info(f"Search results written to {search_results_file}.")

    # Recheck the size of the recalls 
    if len(recalls) == 0:
        logging.warning("No recalls found.")
    else:
        # Count the number of searches
        searches_lists = [result for result in search_result if result['operation'] == 'search']
        logging.info(f"Total recalls: {len(recalls)}, search results: {len(searches_lists)}.")

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
            if operation_type == 'S':
                # Pop one from the recalls
                first = recalls.pop(0)
                f.write(f"{i}\t{operation_type}\t{first:.4f}\t{latency:.6f}\t{qps_moment:.6f}\n")
            else:
                f.write(f"{i}\t{operation_type}\t0.0000\t{latency:.6f}\t{qps_moment:.6f}\n")