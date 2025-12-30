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
# Function to run exact search in a range of indices
def run_exact_search_range(config, tid, start, end, trace):

    conn2 = connect_to_postgres(config)
    cursor2 = conn2.cursor()
    
    limit = 100

    count = 0
    for i in range(start, end + 1):
        query_vector = trace[i]['payload']
        query_string = f"SELECT id, embedding <-> \'{query_vector.tolist()}\' AS _score FROM items ORDER BY _score LIMIT {limit};"

        with conn2.transaction():
            cursor2.execute("SET LOCAL enable_indexscan = off")
            cursor2.execute(query_string)

        searched = cursor2.fetchall()

        # 
        ids = np.array([x[0] for x in searched], dtype=np.int32)
        scores = np.array([x[1] for x in searched], dtype=np.float32)

        count += 1

        if count % 100 == 0:
            print(f"    Reporting thread {tid}: {(count / (end - start)) * 100 :.2f}% done.")

        trace[i]['gt_ids'] = ids
        trace[i]['gt_scores'] = scores

    conn2.close()


# 
# 
def run_exact_search(config, search_range, trace):

    # We run in multi-threaded mode
    # We run it in 12 threads
    import threading
    index_range_list = []

    # This is the minimum number of threads
    # If not applicable to allocate to multiple threads, say too small range, we run in single thread
    # Detect the current maximum number of threads supported by the system
    try:
        import multiprocessing
        thread_num = multiprocessing.cpu_count()
        logging.info(f"Detected {thread_num} CPU cores.")

        # We only use half, since there is pgvector running
        thread_num = thread_num // 2
        print(f"Using {thread_num} threads for exact search.")

    except Exception as e:
        logging.warning(f"Could not detect CPU cores, using default thread number {thread_num}. Error: {e}")

    if search_range[1] - search_range[0] < (thread_num * 100):
        thread_num = 1
        logging.info(f"Search range {search_range} is too small for multi-threading, running in single thread.")

    query_count = search_range[1] - search_range[0]

    for i in range(thread_num):

        start = search_range[0] + i * (query_count // thread_num)
        end = search_range[0] + (i + 1) * (query_count // thread_num) if i < (thread_num - 1) else search_range[1]

        logging.info(f" >> Assigning thread {i} to handle subrange ({start}, {end}) for exact search...")

        index_range_list.append((start, end))

    threads = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        for pair in index_range_list:
            future = executor.submit(run_exact_search_range, config, i, pair[0], pair[1], trace)

            threads.append(future)

        logging.info("Started threads for exact search...")

        # Wait for all threads to finish
        for future in threads:
            future.result()

    logging.info("All threads for exact search finished.")



def find_search_segments(op_list):
    segments = []
    start = None

    for idx, item in enumerate(op_list):
        if item['operation'] == 'search':
            if start is None:
                start = idx
        else:
            if start is not None:
                segments.append((start, idx - 1))
                start = None

    # Handle case where last items are 'search'
    if start is not None:
        segments.append((start, len(op_list) - 1))

    return segments

# 
# 
# 
def make_workloada_trace(config):
    """
    Generate a trace for workload A based on the ground truth.
    
    :param config: Configuration object.
    :param gt_dicts: List of ground truth dictionaries.
    """
    trace_path = config.get('dataset', 'gt_trace', fallback=None)
    if trace_path is None:
        logging.error("No trace path specified in the configuration.")
        raise ValueError("Trace path must be specified in the configuration under 'dataset' section.")

    if os.path.exists(trace_path):
        logging.info(f"Trace file {trace_path} already exists. Skipping generation.")

        # Peeking at some of the trace
        with open(trace_path, 'rb') as f:
            trace = pickle.load(f)
            if len(trace) > 0:
                pass
            else:
                logging.info("Trace is empty.")

        return
    
    # Check if the trace file already exists
    # If a file exist, we overwrite it
    search_path = config.get('dataset', 'search', fallback=None)
    if search_path is None:
        logging.error("No search vectors path specified in the configuration.")
        raise ValueError("Search vectors path must be specified in the configuration under 'dataset' section.")
    
    query_vectors = np.load(search_path)

    # {
    #     'operation': 'search',
    #     'payload': np.array([0, 1, 2], dtype=np.float32),
    #     'gt_ids': np.array([1, 2, 3], dtype=np.int32),
    #     'gt_scores': np.array([0.1, 0.2, 0.3], dtype=np.float32),
    # }

    trace = []
    for vector in query_vectors:
        trace.append({
            'operation': 'search',
            'payload': vector,
            'gt_ids': None,
            'gt_scores': None
        })

    #
    # Save the trace to a file
    # We do the exact search
    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    # Detect ranges, that have the sequential 'search' operation
    segments = find_search_segments(trace)
    for start, end in segments:
        logging.info(f"    Running exact search for segment ({start}, {end})...")
        run_exact_search(config, (start, end), trace)

    # Save the trace to a file
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)

    conn.close()

    # Check if the searched ids are not empty
    for i, item in enumerate(trace):
        if item['gt_ids'] is None or len(item['gt_ids']) == 0:
            logging.error(f"Search operation at index {i} has no ground truth ids. This might indicate an issue with the exact search.")
            raise ValueError(f"Search operation at index {i} has no ground truth ids.")


#
#
#
def make_workloadb_trace(config):
    """
    Generate a trace for workload B based on the ground truth.
    
    :param config: Configuration object.
    """
    trace_path = config.get('dataset', 'gt_trace', fallback=None)
    if trace_path is None:
        logging.error("No trace path specified in the configuration.")
        raise ValueError("Trace path must be specified in the configuration under 'dataset' section.")
    
    if os.path.exists(trace_path):
        logging.info(f"Trace file {trace_path} already exists. Skipping generation.")

        # Peeking at some of the trace
        with open(trace_path, 'rb') as f:
            trace = pickle.load(f)
            if len(trace) > 0:
                # logging.info(f"First operation in the trace: {trace[0]['operation']}, gt_ids: {trace[0]['gt_ids']}, gt_scores: {trace[0]['gt_scores']}")
                pass
            else:
                logging.info("Trace is empty.")

        return
    
    # 
    base_path = config.get('dataset', 'base', fallback=None)
    search_path = config.get('dataset', 'search', fallback=None)

    if search_path is None:
        logging.error("Search, insert or delete vectors path not specified in the configuration.")
        raise ValueError("Search, insert and delete vectors paths must be specified in the configuration under 'dataset' section.")
    
    search_vectors = np.load(search_path)
    split_num = config.getint('dataset', 'split_num', fallback=None)
    if split_num is None:
        logging.error("No split number specified in the configuration.")
        raise ValueError("Split number must be specified in the configuration under 'pgvector' section.")
    

    # Set the base vectors.
    if '*' in base_path:
        # If the base path is a glob pattern, we need to find the first matching file
        # Find all files matching the pattern
        import glob
        import gc
        from pathlib import Path
        
        matching_files = sorted(glob.glob(base_path))
        if not matching_files:
            logging.error(f"No files found matching pattern: {base_path}")
            exit(1)
        
        logging.info(f"Found {len(matching_files)} files matching pattern: {base_path}")
        for f in matching_files:
            logging.info(f"  - {Path(f).name}")

        loaded_vectors = 0
        for file_idx, file_path in enumerate(matching_files):
            logging.info(f"Loading base vectors from file {file_idx + 1}/{len(matching_files)}: {file_path}")
            vectors = np.load(file_path)
            if loaded_vectors == 0:
                base_vectors = vectors
            else:
                base_vectors = np.concatenate((base_vectors, vectors), axis=0)
            
            loaded_vectors += len(vectors)
            logging.info(f"Loaded {loaded_vectors} vectors so far.")

            if loaded_vectors >= split_num * 2:
                logging.info(f"Loaded enough vectors for split: {split_num * 2}. Stopping loading.")
                break
    else:
        # 
        # We always do the split, and generate the 
        base_vectors = np.load(base_path)
    
    if split_num > len(base_vectors) // 2:
        logging.error(f"Split number {split_num} is larger than half of the base vectors length {len(base_vectors) // 2}.")
        raise ValueError("Split number must be less than or equal to half of the base vectors length.")

    first_split = base_vectors[:split_num]
    second_split = base_vectors[split_num:split_num * 2]

    # Randomly select the insert_ratio of the first split
    insert_ratio = config.getfloat('workload', 'insert_ratio', fallback=0.001)
    if insert_ratio < 0 or insert_ratio > 1:
        logging.error(f"Insert ratio {insert_ratio} is out of bounds (0, 1).")
        raise ValueError("Insert ratio must be between 0 and 1.")
    
    insert_num = int(len(first_split) * insert_ratio)
    insert_vectors = second_split[np.random.choice(len(second_split), insert_num, replace=False)]

    # Check the max id
    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(id) FROM items;")
    max_id = cursor.fetchone()[0]

    delete_num = insert_num
    delete_ids = np.random.choice(range(0, max_id), size=delete_num, replace=False)

    # Create the trace
    trace = []
    request_num = len(search_vectors) + insert_num + delete_num

    operations = ['search', 'insert', 'delete']

    # Throw away the 4th digit of the search probability
    search_probability = len(search_vectors) / request_num
    search_probability = round(search_probability, 4)

    insert_probability = (1 - search_probability) / 2
    delete_probability = (1 - search_probability) / 2

    probabilities = [search_probability, insert_probability, delete_probability]

    logging.info(f"Search probability: {search_probability}, Insert probability: {insert_probability}, Delete probability: {delete_probability}")

    search_index = 0
    insert_index = 0
    delete_index = 0

    left_searches = len(search_vectors)
    left_inserts = len(insert_vectors)
    left_deletes = len(delete_ids)

    for i in range(request_num):
        operation = random.choices(operations, weights=probabilities, k=1)[0]
        
        if operation == 'search':
            vector = search_vectors[search_index]
            trace.append({
                'operation': 'search',
                'payload': vector,
                'gt_ids': None,
                'gt_scores': None
            })

            search_index += 1
            left_searches = len(search_vectors) - search_index

        elif operation == 'insert':
            vector = insert_vectors[insert_index]
            trace.append({
                'operation': 'insert',
                'payload': vector,
                'gt_ids': None,
                'gt_scores': None
            })
            insert_index += 1
            left_inserts = len(insert_vectors) - insert_index


        elif operation == 'delete':
            id = delete_ids[delete_index]
            trace.append({
                'operation': 'delete',
                'payload': id,
                'gt_ids': None,
                'gt_scores': None
            })
            delete_index += 1
            left_deletes = len(delete_ids) - delete_index

        total_left = left_searches + left_inserts + left_deletes
        if total_left == 0:
            logging.info("All operations are done.")
            break

        search_probability = round(left_searches / total_left, 4)
        insert_probability = round(left_inserts / total_left, 4)
        delete_probability = round(left_deletes / total_left, 4)

        probabilities = [search_probability, insert_probability, delete_probability]
    
    segments = find_search_segments(trace)
    for start, end in segments:
        logging.info(f"    Running exact search for segment ({start}, {end})...")
        run_exact_search(config, (start, end), trace)

    # Check if the searched ids are not empty
    for i, item in enumerate(trace):
        if item['operation'] == 'search':
            if item['gt_ids'] is None or len(item['gt_ids']) == 0:
                logging.error(f"Search operation at index {i} has no ground truth ids. This might indicate an issue with the exact search.")
                raise ValueError(f"Search operation at index {i} has no ground truth ids.")

    # Save the trace to a file
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)
    
    conn.close()

#
#
#
def make_stress_trace(config):
    """
    Generate a trace for stress testing based on the ground truth.
    
    :param config: Configuration object.
    """
    trace_path = config.get('dataset', 'gt_trace', fallback=None)
    if trace_path is None:
        logging.error("No trace path specified in the configuration.")
        raise ValueError("Trace path must be specified in the configuration under 'dataset' section.")
    
    if os.path.exists(trace_path):
        logging.info(f"Trace file {trace_path} already exists. Skipping generation.")

        # Peeking at some of the trace
        with open(trace_path, 'rb') as f:
            trace = pickle.load(f)
            if len(trace) > 0:
                # logging.info(f"First operation in the trace: {trace[0]['operation']}, gt_ids: {trace[0]['gt_ids']}, gt_scores: {trace[0]['gt_scores']}")
                pass
            else:
                logging.info("Trace is empty.")

        return
    
    # 
    base_path = config.get('dataset', 'base', fallback=None)
    search_path = config.get('dataset', 'search', fallback=None)

    if search_path is None:
        logging.error("Search, insert or delete vectors path not specified in the configuration.")
        raise ValueError("Search, insert and delete vectors paths must be specified in the configuration under 'dataset' section.")
    
    search_vectors = np.load(search_path)
    split_num = config.getint('dataset', 'split_num', fallback=None)
    if split_num is None:
        logging.error("No split number specified in the configuration.")
        raise ValueError("Split number must be specified in the configuration under 'pgvector' section.")
    

    # Set the base vectors.
    if '*' in base_path:
        # If the base path is a glob pattern, we need to find the first matching file
        # Find all files matching the pattern
        import glob
        import gc
        from pathlib import Path
        
        matching_files = sorted(glob.glob(base_path))
        if not matching_files:
            logging.error(f"No files found matching pattern: {base_path}")
            exit(1)
        
        logging.info(f"Found {len(matching_files)} files matching pattern: {base_path}")
        for f in matching_files:
            logging.info(f"  - {Path(f).name}")

        loaded_vectors = 0
        for file_idx, file_path in enumerate(matching_files):
            logging.info(f"Loading base vectors from file {file_idx + 1}/{len(matching_files)}: {file_path}")
            vectors = np.load(file_path)
            if loaded_vectors == 0:
                base_vectors = vectors
            else:
                base_vectors = np.concatenate((base_vectors, vectors), axis=0)
            
            loaded_vectors += len(vectors)
            logging.info(f"Loaded {loaded_vectors} vectors so far.")

            if loaded_vectors >= split_num * 2:
                logging.info(f"Loaded enough vectors for split: {split_num * 2}. Stopping loading.")
                break
    else:
        # 
        # We always do the split, and generate the 
        base_vectors = np.load(base_path)
    
    if split_num > len(base_vectors) // 2:
        logging.error(f"Split number {split_num} is larger than half of the base vectors length {len(base_vectors) // 2}.")
        raise ValueError("Split number must be less than or equal to half of the base vectors length.")

    first_split = base_vectors[:split_num]
    second_split = base_vectors[split_num:split_num * 2]

    # Limit the second_split (inserts) max to 5% (of the first split)
    first_split_len = len(first_split)
    second_split_len = int(0.05 * first_split_len)
    
    logging.info(f"Second split (insert) length: {second_split_len}")
    second_split = second_split[:second_split_len]

    
    # Prepare the full insert vectors as the workload form.
    insert_vectors = second_split

    # Do not check the max id. Insert vectors are set to have -1 id, which will be later determined
    # Create the trace
    trace = []
    request_num = len(search_vectors) + len(insert_vectors)

    operations = ['search', 'insert']

    # First is search, second is the inserts.
    search_trace = []
    for vector in search_vectors:
        search_trace.append({
            'operation': 'search',
            'payload': vector,
            'gt_ids': None,
            'gt_scores': None
        })
    
    insert_trace = []
    for vector in insert_vectors:
        insert_trace.append({
            'operation': 'insert',
            'payload': vector,
            'gt_ids': None,
            'gt_scores': None
        })

    # After preparation, make the ground truth for only after "Insert".
    # Insert counts: len(search vectors)
    conn = connect_to_postgres(config)
    cursor = conn.cursor()

    max_id = 0
    cursor.execute("SELECT MAX(id) FROM items;")
    result = cursor.fetchone()

    logging.info(f"Current max id in the database: {result[0]}")

    if result is not None and result[0] is not None:
        max_id = result[0]

    for i in range(len(insert_vectors)):

        # First, insert the vectors one by one
        vector = insert_trace[i]['payload']
        insert_sql = "INSERT INTO items (id, embedding) VALUES (%s, %s);"

        # Insert a vector
        with conn.transaction():
            cursor.execute(insert_sql, (max_id + 1, vector))
            max_id += 1

        if (i + 1) % 100 == 0:
            logging.info(f"Inserted {i + 1}/{len(insert_trace)} vectors...")

    logging.info("All insertions are done.")
    logging.info("Starting exact search for the search trace...")

    # After bulk insert, we do the exact search
    segments = find_search_segments(search_trace)
    for start, end in segments:
        logging.info(f"    Running exact search for segment ({start}, {end})...")
        run_exact_search(config, (start, end), search_trace)

    trace = {
        "search": search_trace,
        "insert": insert_trace
    }


    conn.close()

    
    # Save the trace to a file
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)


# Create the final workload
# 
# run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth for pgvector benchmark.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the configuration
    config = load_configuration(args.config)

    # Load the workload type
    workload_type = config.get('workload', 'wtype', fallback=None)
    if workload_type is None:
        logging.error("No workload type specified in the configuration.")
        exit(1)
    
    if workload_type == 'workloada':
        logging.info("Running workload A trace generation...")
        make_workloada_trace(config)
    elif workload_type == 'workloadb':
        logging.info("Running workload B trace generation...")
        make_workloadb_trace(config)
    # Does the workload type starts with stress?
    elif workload_type.startswith('stress'):
        logging.info("Running stress test trace generation...")
        make_stress_trace(config)
    else:
        logging.error(f"Unknown workload type: {workload_type}. Supported types are 'workloada' and 'workloadb'.")
        exit(1)
