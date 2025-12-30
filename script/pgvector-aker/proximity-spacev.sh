#!/bin/bash

# Check if the current root working directory is set.
# If not, print the error message and exit.
if [ -z "${WORKING_PATH}" ]; then
    printf "WORKING_PATH is not set. Please run activate.sh first.\n"
    exit 1
fi

export PATH=/usr/local/pgsql/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/pgsql/lib:$LD_LIBRARY_PATH
export MANPATH=/usr/local/pgsql/share/man:$MANPATH

export CPLUS_INCLUDE_PATH=/usr/local/pgsql/include
export C_INCLUDE_PATH=/usr/local/pgsql/include

# Compile:
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

working_path="$WORKING_PATH/script/pgvector-aker"
printf "[${CYAN}INFO${NC}] Starting the pgvector experiment script...\n"

# Activate the virtual environment
# Initialize the venv, if not found.
if [ ! -d "$working_path/venv" ]; then
    printf "Virtual environment not found. Initializing...\n"
    python3 -m venv "$working_path/venv"
    if [ $? -ne 0 ]; then
        printf "Failed to create virtual environment. Please check your Python installation.\n"
        exit 1
    fi

    printf "Virtual environment created successfully.\n"
    source "$working_path/venv/bin/activate"

    # Install necessary packages
    requirements=(
        "stopit" "setuptools" "typer" "numpy" "pyarrow" "tqdm" "pgvector" "psycopg"
    )

    for requirement in "${requirements[@]}"; do
        pip install "$requirement"
    done
    printf "Python dependencies installed successfully.\n"

else
    printf "Virtual environment found. Activating...\n"
    source "$working_path/venv/bin/activate"
fi



# List all the files in the configuratino directory, and grab all the .ini files.
config_files=$(find -L "$working_path/0-conf" -type f -name "*.ini")

printf "Configuration files found:\n%s\n" "$config_files"

# export PG_CONFIG=/usr/local/pgsql/bin/pg_config
# make -j > /dev/null 2>&1
# sudo --preserve-env=PG_CONFIG make install

cd ${working_path}

#
# Now, we are ready to run the experiments.
# We loop through each configuration file, and run the experiment.

topkache_config_path="$working_path/topkache-params.json"
export TOPKACHE_CONFIG=${topkache_config_path}

printf "[${CYAN}INFO${NC}] TopKache configuration path: %s\n" "$topkache_config_path"

alpha_pairs=(
    "0.25 1.1"
)

vector_pool_size=(
    100000 200000 500000 1000000 2000000 3000000
)

fixed_threshold=(
    9961 7575 9940 9450
)

start_threshold=(
    0.5
)

write_topkache_config() {
    local config_path=$1
    local vector_pool_size_value=$2
    local fixed_threshold_value=$3
    local start_threshold_value=$4
    local alpha_tighten_value=$5
    local alpha_loosen_value=$6

    # Extract the distance metric
    local distance_metric_value=$(grep -E '^distance\s*=' "$config_path" | cut -d '=' -f2 | xargs)
    local dimension_value=$(grep -E '^dim\s*=' "$config_path" | cut -d '=' -f2 | xargs)
    local limit=$(grep -E '^limit\s*=' "$config_path" | cut -d '=' -f2 | xargs)

    # We only take two for now:
    # If the value is "vector_l2_ops", we set to L2.
    if [ "$distance_metric_value" = "vector_l2_ops" ]; then
        distance_metric_value="L2"

    elif [ "$distance_metric_value" = "vector_ip_ops" ]; then
        distance_metric_value="IP"

    else
        printf "[${YELLOW}WARNING${NC}] Unknown distance metric: %s. Defaulting to L2.\n" \
            "$distance_metric_value"
        distance_metric_value="UNKNOWN"
    fi

    printf "[${CYAN}INFO${NC}] Writing TopKache configuration to %s\n" "$topkache_config_path"

    echo "{
        \"vector_dim\": ${dimension_value},
        \"vector_pool_size\": ${vector_pool_size_value},
        \"vector_list_size\": ${limit},
        \"vector_data_size\": $((4 * dimension_value)),
        \"vector_intopk\": ${limit},
        \"vector_extras\": 0,
        \"similar_match\": 1,
        \"fixed_threshold\": ${fixed_threshold_value},
        \"start_threshold\": ${start_threshold_value},
        \"risk_threshold\": 0.3,
        \"alpha_tighten\": ${alpha_tighten_value},
        \"alpha_loosen\": ${alpha_loosen_value},
        \"distance_metric\": \"${distance_metric_value}\"
    }" > ${topkache_config_path}

    if [ $? -ne 0 ]; then
        printf "[${YELLOW}WARNING${NC}] Failed to write TopKache configuration to %s\n" "$topkache_config_path"
        exit 1
    else
        printf "[${CYAN}INFO${NC}] TopKache configuration written successfully.\n"
    fi
}

for config_file in $config_files; do

    for vector_pool_size_value in "${vector_pool_size[@]}"; do
        for fixed_threshold_value in "${fixed_threshold[@]}"; do
            for start_threshold_value in "${start_threshold[@]}"; do
                for pair in "${alpha_pairs[@]}"; do
                    read -r alpha_tighten_value alpha_loosen_value <<< "$pair"
                # for alpha_tighten_value in "${alpha_tighten[@]}"; do
                #     for alpha_loosen_value in "${alpha_loosen[@]}"; do

                        if [ ! -f "$config_file" ]; then
                            printf "[${YELLOW}WARNING${NC}] Configuration file %s not found. Skipping...\n" "$config_file"
                            continue
                        fi

                        datastore=$(grep -E '^datastore\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        host=$(grep -E '^host\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        port=$(grep -E '^port\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        user=$(grep -E '^user\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        password=$(grep -E '^password\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        dbname=$(grep -E '^dbname\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        psql_config=$(grep -E '^psql_config\s*=' "$config_file" | cut -d '=' -f2 | xargs)

                        workload_name=$(grep -E '^name\s*=' "$config_file" | cut -d '=' -f2 | xargs)
                        workload_type=$(grep -E '^wtype\s*=' "$config_file" | cut -d '=' -f2 | xargs)

                        write_topkache_config \
                            "$config_file" \
                            "$vector_pool_size_value" \
                            "$fixed_threshold_value" \
                            "$start_threshold_value" \
                            "$alpha_tighten_value" \
                            "$alpha_loosen_value"

                        cat ${topkache_config_path}
                        

                        # 
                        # When running, we assume that all necessary index and data are already built and created.
                        # If not, we skip the run.
                        # So, before we run, we check if 
                        # 1) If the clean database exists
                        # 2) If the index exists
                        # 3) If the data exists

                        printf "[${CYAN}INFO${NC}] Starting workload: %s\n" "$workload_name"

                        datastore_path=${working_path}/${datastore}
                        if [ -d "${datastore_path}-clean" ]; then

                            printf "[${YELLOW}INFO${NC}] Clean datastore path found: %s. Recovering...\n" \
                                "${datastore_path}-clean"
                            printf "Workload type: %s\n" "$workload_type"
                                    
                            rm -rf "${datastore_path}"
                            cp -r "${datastore_path}-clean" "${datastore_path}"
                            sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

                        else
                            printf "[${YELLOW}WARNING${NC}] Clean datastore path %s not found. To run this script, it needs a clean backup. Skipping the workload run...\n" \
                                "${datastore_path}-clean"
                            continue
                        fi

                        # Before we do the experiment, we count the number of requests by reading the pkl file.
                        trace_file=$(grep -E '^gt_trace\s*=' "$config_file" | cut -d '=' -f2 | xargs)

                        # Since the PostgreSQL automatically vacuums (deal with the deletes), 
                        # we only consider the request count the 'search' and 'insert' requests. 
                        # Exclude the count if the operation type is 'delete'.
                        # requests_count=$(python3 -c "import pickle; f=open('${trace_file}','rb'); data=pickle.load(f); print(len([req for req in data if req['operation'] != 'delete'])); f.close()")
                        requests_count=$(python3 -c "import pickle; f=open('${trace_file}','rb'); data = pickle.load(f); print(len(data))")

                        printf "[${CYAN}INFO${NC}] Number of requests (excluding deletes) in the trace file %s: %d\n" \
                            "$trace_file" "$requests_count"

                        export TOPKACHE_REQS=${requests_count}
                        export TOPKACHE_TRACE_DIR=topkache-trace-${workload_name}-${vector_pool_size_value}-${fixed_threshold_value}-${start_threshold_value}-${alpha_tighten_value}-${alpha_loosen_value}

                        # We run the server
                        numactl --cpunodebind=0 --membind=0 pg_ctl -D "${datastore_path}" \
                            -l ${datastore_path}/logs/pg-$(date +"%Y-%m-%d_%H-%M-%S").log \
                            -o "--config-file=${working_path}/${psql_config}" \
                            -o "-p ${port}" \
                            start

                        sleep 0.5
                        pg_isready -h "$host" -p "$port" -U "$user"
                        if [ $? -ne 0 ]; then
                            printf "PostgreSQL is not ready. Please check the logs.\n"

                            numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} stop -m immediate
                            exit 1
                        fi

                        # Running the index check
                        python3 "$working_path/2-runs/dscheck.py" --config "$config_file"

                        # Running the script: The trace file should be generated first.
                        numactl --cpunodebind=0 --membind=0 python3 "$working_path/2-runs/run-2.py" --config "$config_file"
                        sleep 60

                        result_file_dir="$working_path/4-results/proximity-spacev/${workload_name}/${vector_pool_size_value}-${fixed_threshold_value}-${start_threshold_value}-${alpha_tighten_value}-${alpha_loosen_value}"
                        mkdir -p "$result_file_dir"

                        mv report.csv "$result_file_dir/report.csv"
                        mv search-results.pkl "$result_file_dir/search-results.pkl"
                        mv trace-extract-info.csv "${result_file_dir}/trace-extract-info.csv"

                        mv /tmp/${TOPKACHE_TRACE_DIR} ${result_file_dir}/
                        rm -rf ${TOPKACHE_TRACE_DIR}

                        # Move the log file also.
                        mv ${datastore_path}/log/*.log ${result_file_dir}/

                        printf "[${CYAN}PYTHON${NC}] Workload %s completed. Results saved in %s\n" \
                            "$workload_name" "$result_file_dir"

                        # Stop the PostgreSQL server
                        numactl --cpunodebind=0 --membind=0 pg_ctl -D "${datastore_path}" stop -m immediate
                        sleep 0.5

                #     done
                # done
                done
            done
        done
    done
done

rm ${topkache_config_path}
