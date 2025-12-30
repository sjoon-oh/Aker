#!/bin/bash

export PATH=/usr/local/pgsql/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/pgsql/lib:$LD_LIBRARY_PATH
export MANPATH=/usr/local/pgsql/share/man:$MANPATH

export CPLUS_INCLUDE_PATH=/usr/local/pgsql/include
export C_INCLUDE_PATH=/usr/local/pgsql/include

postgres_data_dir=$1
initialization_sql_file=$2

# Check if the first parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <postgres_data_dir> <initialization_sql_file>"
  exit 1
fi

# Check if Postgres is running
if pgrep -x "postgres" > /dev/null; then
  echo "PostgreSQL server is already running. Stopping it first."

  # Kill the PostgreSQL server if it's running
  kill -SIGINT $(pgrep postgres | sort -n | head -1)
fi

sleep 2

# Check if the directory exists
# If exists, reset it
if [ -d "${postgres_data_dir}" ]; then
  echo "Directory ${postgres_data_dir} already exists. Resetting it."
  rm -rf "${postgres_data_dir}"/*
else
  echo "Creating directory ${postgres_data_dir}."
fi

printf "Initializing PostgreSQL database in %s\n" "${postgres_data_dir}"
initdb -D "${postgres_data_dir}" -U postgres
sleep 1

# Make log directory
# mkdir -p "${postgres_data_dir}/log"

printf "Starting PostgreSQL server in %s\n" "${postgres_data_dir}"
printf "logging to log/log-<timestamp>.log\n"
mkdir -p log

log_timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
config_file=$3

pg_ctl \
    -D "${postgres_data_dir}" \
    -l log/log-${log_timestamp}.log \
    -o "--config-file=${config_file}" \
    start
printf "Waiting for PostgreSQL server to start\n"

sleep 3
pg_isready -h localhost -p 5432 -U postgres
if [ $? -ne 0 ]; then
  echo "PostgreSQL server did not start successfully."
  exit 1
fi

#
# Initialize the database
printf "Creating initial database and user\n"
PGASSWORD=passwd psql -h localhost -p 5432 -U postgres -f "${initialization_sql_file}"
if [ $? -ne 0 ]; then
  echo "Failed to initialize the database."
  exit 1
fi

# Stopping PostgreSQL server
printf "Stopping PostgreSQL server\n"
pg_ctl -D "${postgres_data_dir}" stop