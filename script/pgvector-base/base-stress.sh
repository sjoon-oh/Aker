#!/bin/bash

# Check if the current root working directory is set.
# If not, print the error message and exit.
if [ -z "${WORKING_PATH}" ]; then
    printf "WORKING_PATH is not set. Please run activate.sh first.\n"
    exit 1
fi

pkill postgres

# Arguments: 
# --skip-init: Skip the index build, and only run the search.
# --skip-search: Skip the search, and only run the index build.

# Parse the arguments
skip_init=false
skip_runs=false
skip_build=false

skip_pgvector_build=true
skip_base_all=false
custom_script=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-init)
            skip_init=true
            shift # Remove the argument from the list
            ;;
        --skip-runs)
            skip_runs=true
            shift # Remove the argument from the list
            ;;
        --skip-build)
            if [ "$skip_init" = true ]; then
                printf "Cannot skip build without skipping init. Please use --skip-init.\n"
                exit 1
            fi
            skip_build=true
            shift # Remove the argument from the list
            ;;
        --pgvector-build)
            skip_pgvector_build=false
            shift # Remove the argument from the list
            ;;
        --skip-base-all)
            skip_base_all=true
            shift # Remove the argument from the list
            ;;
        --custom-script)
            if [ -z "$2" ]; then
                printf "No custom script provided after --custom-script.\n"
                exit 1
            fi
            custom_script+=("$2") # Add the custom script to the array
            shift 2 # Remove the argument and its value from the list
            ;;
        *)
            printf "Unknown argument: %s\n" "$1"
            printf "Usage: $0 [--skip-init]\n"
            exit 1
            ;;
    esac
done

export PATH=/usr/local/pgsql/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/pgsql/lib:$LD_LIBRARY_PATH
export MANPATH=/usr/local/pgsql/share/man:$MANPATH

export CPLUS_INCLUDE_PATH=/usr/local/pgsql/include
export C_INCLUDE_PATH=/usr/local/pgsql/include

# Compile:
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 
# Set the working path for the script.
working_path=$WORKING_PATH/script/pgvector-base
cd ${working_path}

scratchpad_path="$working_path/3-scratchpad"

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

if [ "$skip_base_all" = false ]; then
    printf "[${YELLOW}INFO${NC}] Starting the base runs...\n"

    # List all the files in the configuratino directory, and grab all the .ini files.
    config_files=$(find -L "$working_path/0-conf" -type f -name "*.ini")

    printf "Configuration files found:\n%s\n" "$config_files"

    # # Reset the scratchpad path.
    # if [ -d "$scratchpad_path" ]; then
    #     printf "Scratchpad path already exists. Removing it: %s\n" "$scratchpad_path"
    #     rm -rf "$scratchpad_path"
    # fi

    # mkdir -p "$scratchpad_path"
    # if [ ! -d "$scratchpad_path/pgvector" ]; then
    #     pgvector_source_path=$WORKING_PATH/apps/pgvector/pgvector
    #     printf "Linking pgvector source directory to scratchpad: %s\n" \
    #         ${pgvector_source_path}

    #     # Link the source path to toeh scratchpad path.
    #     ln -s "$pgvector_source_path" "$scratchpad_path/pgvector"
    # fi

    # 
    # If not skip_pgvector_build, we compile the pgvector extension.
    # if [ "$skip_pgvector_build" = false ]; then

    #     printf "[${YELLOW}INFO${NC}] Starting the compilation of pgvector extension...\n"
    #     cd "$scratchpad_path/pgvector"

    #     git reset --hard HEAD
    #     make clean

    #     export PG_CONFIG=/usr/local/pgsql/bin/pg_config
    #     make -j > /dev/null 2>&1
    #     if [ $? -ne 0 ]; then
    #         printf "Failed to compile pgvector extension. Please check the logs.\n"
    #         exit 1
    #     fi

    #     sudo --preserve-env=PG_CONFIG make install > /dev/null 2>&1
    # fi

    # sudo --preserve-env=PG_CONFIG make install
    # printf "pgvector extension clean installed successfully.\n"

    cd ${working_path}


    # Extract the datastore path, from the configuration file
    for config_file in $config_files; do

        printf "[${YELLOW}INFO${NC}] Processing configuration file: %s\n" "$config_file"

        datastore=$(grep -E '^datastore\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        host=$(grep -E '^host\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        port=$(grep -E '^port\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        user=$(grep -E '^user\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        password=$(grep -E '^password\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        dbname=$(grep -E '^dbname\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        psql_config=$(grep -E '^psql_config\s*=' "$config_file" | cut -d '=' -f2 | xargs)

        workload_name=$(grep -E '^name\s*=' "$config_file" | cut -d '=' -f2 | xargs)
        workload_type=$(grep -E '^wtype\s*=' "$config_file" | cut -d '=' -f2 | xargs)

        if [ -n "$datastore" ]; then
            datastore_path=${working_path}/${datastore}
            printf "Datastore path extracted from %s: %s\n" "$config_file" "$datastore_path"

            # If not skipping.
            if [ "$skip_init" = false ]; then

                # Check if the clean datastore path exists.
                if [ -d "${datastore_path}-clean" ]; then
                    # Run the recovery
                    printf "[${YELLOW}INFO${NC}] Clean datastore path found: %s. Recovering...\n" \
                        "${datastore_path}-clean"
                    printf "Workload type: %s\n" "$workload_type"

                    # If workload type is workloada, we skip the recovery.
                    # rm -rf "${datastore_path}"
                    # cp -r "${datastore_path}-clean" "${datastore_path}"
                    rm -rf ${datastore_path}/log/*
                    rm -rf ${datastore_path}/logs/*

                    sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

                    skip_build=true

                    numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} \
                        -l ${datastore_path}/logs/pg-$(date +"%Y-%m-%d_%H-%M-%S").log \
                        -o "--config-file=${working_path}/${psql_config}" \
                        -o "-p ${port}" \
                        start

                    sleep 1
                    pg_isready -h "$host" -p "$port" -U "$user"
                    if [ $? -ne 0 ]; then
                        printf "PostgreSQL is not ready (skipping init). Please check the logs.\n"

                        numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} stop
                        exit 1
                    fi

                else

                    initdb -D "${datastore_path}" -U "$user"

                    printf "PostgreSQL database initialized in %s\n" "${datastore_path}"
                    mkdir -p "${datastore_path}/logs"

                    numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} \
                        -l ${datastore_path}/logs/pg-$(date +"%Y-%m-%d_%H-%M-%S").log \
                        -o "--config-file=${working_path}/${psql_config}" \
                        -o "-p ${port}" \
                        start

                    sleep 0.5
                    pg_isready -h "$host" -p "$port" -U "$user"
                    if [ $? -ne 0 ]; then
                        printf "PostgreSQL is not ready. Please check the logs.\n"

                        numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} stop
                        exit 1
                    fi

                    printf "[${CYAN}PYTHON${NC}] Creating pgvector extension in database %s...\n" "$dbname"
                    python3 "$working_path/2-runs/create.py" --config "$config_file"

                    skip_build=false
                fi
            fi

            if [ "$skip_build" = false ]; then
                printf "[${CYAN}PYTHON${NC}] Uploading data in database %s...\n" "$dbname"
                python3 "$working_path/2-runs/upload.py" --config "$config_file"

                psql "host=$host port=$port user=$user password=$password dbname=$dbname" -c \
                    "ALTER SYSTEM SET max_worker_processes = 100;"
                numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} \
                    -l ${datastore_path}/logs/pg-$(date +"%Y-%m-%d_%H-%M-%S").log \
                    -o "--config-file=${working_path}/${psql_config}" \
                    -o "-p ${port}" \
                    restart

                # Building the index
                printf "[${CYAN}PYTHON${NC}] Building the index in database %s...\n" "$dbname"
                python3 "$working_path/2-runs/build.py" --config "$config_file"                

                # Check if the index exists.
                printf "[${CYAN}PYTHON${NC}] Checking the index in database %s...\n" "$dbname"
                python3 "$working_path/2-runs/dscheck.py" --config "$config_file"

                # Now, we create the backup.
                printf "[${CYAN}PYTHON${NC}] Creating a backup of the database %s...\n" "$dbname"

                # First stop the database.
                numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} stop
                cp -r "${datastore_path}" "${datastore_path}-clean"

                # Restart the database.
                printf "[${CYAN}PYTHON${NC}] Restarting the database %s...\n" "$dbname"
                numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} \
                    -l ${datastore_path}/logs/pg-$(date +"%Y-%m-%d_%H-%M-%S").log \
                    -o "--config-file=${working_path}/${psql_config}" \
                    -o "-p ${port}" \
                    start
                sleep 0.5

                pg_isready -h "$host" -p "$port" -U "$user"
                if [ $? -ne 0 ]; then
                    printf "PostgreSQL is not ready. Please check the logs.\n"
                    exit 1
                fi

            else
                printf "[${CYAN}PYTHON${NC}] Skipping the index build in database %s...\n" "$dbname"

                # Check if the index exists.
                python3 "$working_path/2-runs/dscheck.py" --config "$config_file"

                # Check if the backup exists.
                if [ ! -d "${datastore_path}-clean" ]; then
                    printf "[${YELLOW}WARNING${NC}] Backup not found. Please run the index build first.\n"
                    exit 1
                else
                    printf "[${CYAN}PYTHON${NC}] Backup found: %s\n" "${datastore_path}-clean"
                fi
            fi

            #
            # We are ready to run the workload.
            if [ "$skip_runs" = false ]; then
                printf "[${CYAN}PYTHON${NC}] Running the workload %s in database %s...\n" \
                    "$workload_name" "$dbname"

                # trace.py skips generation if the trace file already exists.
                # So, we run it first.
                python3 "$working_path/2-runs/trace-stress.py" --config "$config_file"

                # Next, the workload trace may have some search gt holes, so we fix it.
                python3 "$working_path/2-runs/fix.py" --config "$config_file"
                numactl --cpunodebind=0 --membind=0 python3 "$working_path/2-runs/run-2.py" --config "$config_file"

                # Move the result file to to the results directory.
                result_file_dir="$working_path/4-results/${workload_name}"

                mkdir -p "$result_file_dir"
                mv report.csv "$result_file_dir/report.csv"
                mv search-results.pkl "$result_file_dir/search-results.pkl"
                mv trace-extract-info.csv "${result_file_dir}/trace-extract-info.csv"

                # Move the log file also.
                mv ${datastore_path}/log/*.log $result_file_dir/

                printf "[${CYAN}PYTHON${NC}] Workload %s completed. Results saved in %s/report.csv\n" \
                    "$workload_name" "$result_file_dir"
            else
                printf "[${YELLOW}WARNING${NC}] Skipping the workload run in database %s...\n" "$dbname"
            fi

            numactl --cpunodebind=0 --membind=0 pg_ctl -D ${datastore_path} stop
        else
            printf "No datastore path found in %s. Skipping...\n" "$config_file"
        fi
    done

    # For all the result files, we merge them into a single file
    printf "[${YELLOW}INFO${NC}] Merging the result files...\n"
    result_files=$(find "$working_path/4-results" -type f -name "report.csv")
    merged_result_file="$working_path/4-results/base-report.csv"

    # Write the header to the merged file
    echo -e "Workload\tType\tSearch Params\tQPS\tAvg Latency(ms)\t50%ile Latency(ms)\t99%ile Latency(ms)\tAvg Recall" 
        > "$merged_result_file"

    # Loop through each result file and append its content to the merged file
    for result_file in $result_files; do
        if [ -f "$result_file" ]; then
            # Get the second line only
            workload_name=$(basename "$(dirname "$result_file")")
            second_line=$(sed -n '2p' "$result_file")

            write_line=$(echo -e "${workload_name}\t${second_line}")
            echo -e "$write_line" >> "$merged_result_file"
        else
            printf "[${YELLOW}WARNING${NC}] Result file %s not found. Skipping...\n" "$result_file"
        fi
    done

    printf "[${CYAN}INFO${NC}] Base runs completed. Results saved in %s\n" "$merged_result_file"

else
    printf "[${CYAN}INFO${NC}] Skipping the whole base as requested.\n"
fi
