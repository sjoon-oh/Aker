#!/bin/bash

set -euo pipefail

#
# Helper utilities to reproduce legacy experiment safeguards:
# - restore a clean PGDATA snapshot (<datastore_clean> -> <datastore>) when configured
# - ALWAYS drop OS page cache for each benchmark run (legacy behavior)
# - start/stop PostgreSQL using pg_ctl when configured
# - export environment variables from INI so postgres can read them via getenv
# - collect new /tmp trace directories produced during a benchmark run
#
# This file is meant to be sourced by the sample run scripts.
#

BENCH_ROOT="$(pwd)"

resolve_config_path() {
    #
    # Resolve a config path.
    # - If the argument has no '/' and the file exists under configs/, use that.
    # - Otherwise, treat it as a relative/absolute path as provided.
    #
    local config_arg="$1"

    if [[ -z "${config_arg}" ]]; then
        printf ""
        return 0
    fi

    if [[ "${config_arg}" == /* ]]; then
        printf "%s" "${config_arg}"
        return 0
    fi

    if [[ "${config_arg}" != */* ]]; then
        local candidate="configs/${config_arg}"
        if [[ -f "${candidate}" ]]; then
            printf "%s" "${candidate}"
            return 0
        fi
    fi

    printf "%s" "${config_arg}"
}

resolve_path_under_root() {
    #
    # Resolve a path relative to the current working directory.
    #
    local path_arg="$1"

    if [[ -z "${path_arg}" ]]; then
        printf ""
        return 0
    fi

    if [[ "${path_arg}" == /* ]]; then
        printf "%s" "${path_arg}"
        return 0
    fi

    printf "%s" "${BENCH_ROOT}/${path_arg}"
}

resolve_datastore_path() {
    #
    # Resolve a postgres.datastore-like value into an absolute path.
    #
    # Convention:
    # - If value contains '/', it is treated as a relative path under BENCH_ROOT (unless absolute).
    # - Otherwise, it is treated as a datastore name under BENCH_ROOT/datastore/<name>.
    #
    local value="$1"

    if [[ -z "${value}" ]]; then
        printf ""
        return 0
    fi

    if [[ "${value}" == /* ]]; then
        printf "%s" "${value}"
        return 0
    fi

    if [[ "${value}" == */* ]]; then
        printf "%s" "${BENCH_ROOT}/${value}"
        return 0
    fi

    printf "%s" "${BENCH_ROOT}/datastore/${value}"
}

ini_get_value() {
    #
    # Read "key = value" from an INI file under a given section.
    # - Comments starting with '#' or ';' are ignored.
    # - Leading/trailing whitespace is trimmed.
    #
    # Args:
    #   $1: ini_path
    #   $2: section_name (without brackets)
    #   $3: key
    #
    local ini_path="$1"
    local section_name="$2"
    local key="$3"

    awk -v section="["section_name"]" -v key="$key" '
        function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
        function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
        function trim(s)  { return rtrim(ltrim(s)) }

        BEGIN { in_section = 0 }

        {
            line = $0
            sub(/[;#].*$/, "", line)
            line = trim(line)
            if (line == "") { next }

            if (line ~ /^\[/) {
                in_section = (line == section)
                next
            }

            if (!in_section) { next }

            # Match: key = value
            pattern = "^" key "[ \t]*="
            if (line ~ pattern) {
                sub(pattern, "", line)
                line = trim(line)
                print line
                exit
            }
        }
    ' "${ini_path}"
}

ini_list_section_kv() {
    #
    # Print key/value pairs from a section.
    # Output format: "key\tvalue" (tab separated)
    #
    # Args:
    #   $1: ini_path
    #   $2: section_name
    #
    local ini_path="$1"
    local section_name="$2"

    awk -v section="["section_name"]" '
        function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
        function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
        function trim(s)  { return rtrim(ltrim(s)) }

        BEGIN { in_section = 0 }

        {
            line = $0
            sub(/[;#].*$/, "", line)
            line = trim(line)
            if (line == "") { next }

            if (line ~ /^\[/) {
                in_section = (line == section)
                next
            }

            if (!in_section) { next }

            # Match: key = value
            if (match(line, /^[A-Za-z_][A-Za-z0-9_]*[ \t]*=/)) {
                key = substr(line, 1, RLENGTH)
                sub(/[ \t]*=$/, "", key)
                sub(/^[A-Za-z_][A-Za-z0-9_]*[ \t]*=/, "", line)
                value = trim(line)
                print trim(key) "\t" value
            }
        }
    ' "${ini_path}"
}

log_info() {
    printf "[INFO] %s\n" "$*" >&2
}

log_warn() {
    printf "[WARN] %s\n" "$*" >&2
}

log_err() {
    printf "[ERROR] %s\n" "$*" >&2
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log_err "Required command not found in PATH: ${cmd}"
        exit 1
    fi
}

export_env_from_ini() {
    #
    # Export environment variables under [env] section so PostgreSQL can read them via getenv.
    # Keys must be valid shell variable names.
    #
    # Example:
    #   [env]
    #   AKER_CONFIG_PATH = /path/to/aker.json
    #   TOPKACHE_CONFIG = /path/to/topkache.json
    #
    # Args:
    #   $1: config_path
    #   $2: optional AKER_CONFIG_PATH override (wins over INI)
    #
    local config_path="$1"
    local aker_config_override="${2:-}"

    while IFS=$'\t' read -r key value; do
        if [[ -z "${key}" ]]; then
            continue
        fi
        export "${key}=${value}"
        log_info "Exported env: ${key}=${value}"
    done < <(ini_list_section_kv "${config_path}" "env" || true)

    if [[ -n "${aker_config_override}" ]]; then
        local resolved
        resolved="$(resolve_path_under_root "${aker_config_override}")"
        export AKER_CONFIG_PATH="${resolved}"
        log_info "Exported env override: AKER_CONFIG_PATH=${resolved}"
    fi
}

pgctl_stop() {
    local datastore_path="$1"

    require_cmd pg_ctl

    if [[ ! -d "${datastore_path}" ]]; then
        log_warn "datastore path not found; skip pg_ctl stop: ${datastore_path}"
        return 0
    fi

    # Stop only the instance using this data directory.
    pg_ctl -D "${datastore_path}" -m fast stop >/dev/null 2>&1 || true
}

pgctl_start() {
    local datastore_path="$1"
    local psql_config_path="$2"
    local pg_log_path="$3"

    require_cmd pg_ctl

    if [[ ! -d "${datastore_path}" ]]; then
        log_err "datastore path not found; cannot pg_ctl start: ${datastore_path}"
        exit 1
    fi

    mkdir -p "$(dirname -- "${pg_log_path}")"

    if [[ -n "${psql_config_path}" ]]; then
        pg_ctl -D "${datastore_path}" -l "${pg_log_path}" -o "--config-file=${psql_config_path}" start >/dev/null
    else
        pg_ctl -D "${datastore_path}" -l "${pg_log_path}" start >/dev/null
    fi
}

pg_wait_ready() {
    local host="$1"
    local port="$2"
    local user="$3"

    require_cmd pg_isready

    local max_tries=60
    local i=0
    while [[ ${i} -lt ${max_tries} ]]; do
        if pg_isready -h "${host}" -p "${port}" -U "${user}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done

    log_err "PostgreSQL did not become ready (host=${host}, port=${port})"
    return 1
}

drop_os_page_cache() {
    #
    # Legacy requirement: always drop OS page cache for each benchmark run.
    # Prompts for sudo password are allowed.
    #
    log_info "Dropping OS page cache (legacy behavior)"

    if [[ "$(id -u)" -eq 0 ]]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        return 0
    fi

    require_cmd sudo

    # This may prompt for a password; that is intended.
    sudo sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

ensure_clean_snapshot() {
    #
    # Create <datastore_clean_path> if missing by copying <datastore_path>.
    # This can be expensive on large datasets.
    #
    # Args:
    #   $1: config_path (for exporting [env] before restarting postgres)
    #   $2: datastore_path
    #   $3: datastore_clean_path
    #   $4: psql_config_path
    #   $5: pg_log_path
    #   $6: host
    #   $7: port
    #   $8: user
    #
    local config_path="$1"
    local datastore_path="$2"
    local datastore_clean_path="$3"
    local psql_config_path="$4"
    local pg_log_path="$5"
    local host="$6"
    local port="$7"
    local user="$8"
    local aker_config_override="${9:-}"

    if [[ -d "${datastore_clean_path}" ]]; then
        return 0
    fi

    if [[ "${AK_BENCH_CREATE_CLEAN_SNAPSHOT_IF_MISSING:-0}" != "1" ]]; then
        log_err "Clean snapshot not found: ${datastore_clean_path}"
        log_err "Set AK_BENCH_CREATE_CLEAN_SNAPSHOT_IF_MISSING=1 to create it automatically."
        exit 1
    fi

    log_info "Creating clean snapshot: ${datastore_clean_path}"

    pgctl_stop "${datastore_path}"

    if [[ ! -d "${datastore_path}" ]]; then
        log_err "Cannot create clean snapshot; datastore does not exist: ${datastore_path}"
        exit 1
    fi

    rm -rf "${datastore_clean_path}"
    cp -r "${datastore_path}" "${datastore_clean_path}"

    # Bring the server back up for subsequent steps that query PostgreSQL.
    export_env_from_ini "${config_path}" "${aker_config_override}"

    pgctl_start "${datastore_path}" "${psql_config_path}" "${pg_log_path}"
    pg_wait_ready "${host}" "${port}" "${user}"
}

restore_clean_snapshot_and_start() {
    #
    # If postgres.datastore is set:
    # - ensure a clean snapshot exists
    # - restore <datastore_clean> -> <datastore>
    # - ALWAYS drop OS caches
    # - export env vars from [env]
    # - start PostgreSQL from the restored data directory
    #
    local config_path="$1"
    local purpose="$2"  # for log text only
    local aker_config_override="${3:-}"

    local datastore_path
    datastore_path="$(ini_get_value "${config_path}" "postgres" "datastore")"
    datastore_path="$(resolve_datastore_path "${datastore_path}")"
    if [[ -z "${datastore_path}" ]]; then
        # No PGDATA configured; caller likely manages Postgres externally.
        return 0
    fi

    local datastore_clean_path
    datastore_clean_path="$(ini_get_value "${config_path}" "postgres" "datastore_clean")"
    if [[ -z "${datastore_clean_path}" ]]; then
        datastore_clean_path="${datastore_path}-clean"
    fi

    datastore_clean_path="$(resolve_datastore_path "${datastore_clean_path}")"

    local psql_config_path
    psql_config_path="$(ini_get_value "${config_path}" "postgres" "psql_config")"

    if [[ -z "${psql_config_path}" && -f "${BENCH_ROOT}/configs/postgresql.conf" ]]; then
        psql_config_path="configs/postgresql.conf"
    fi

    psql_config_path="$(resolve_path_under_root "${psql_config_path}")"

    local pg_log_path
    pg_log_path="$(ini_get_value "${config_path}" "postgres" "pg_log")"
    if [[ -z "${pg_log_path}" ]]; then
        pg_log_path="output/postgres.log"
    fi
    pg_log_path="$(resolve_path_under_root "${pg_log_path}")"

    local host
    host="$(ini_get_value "${config_path}" "postgres" "host")"
    local port
    port="$(ini_get_value "${config_path}" "postgres" "port")"
    local user
    user="$(ini_get_value "${config_path}" "postgres" "user")"

    # Ensure a clean snapshot exists (optionally create it).
    ensure_clean_snapshot \
        "${config_path}" \
        "${datastore_path}" \
        "${datastore_clean_path}" \
        "${psql_config_path}" \
        "${pg_log_path}" \
        "${host}" \
        "${port}" \
        "${user}" \
        "${aker_config_override}"

    log_info "Restoring clean snapshot for ${purpose}: ${datastore_clean_path} -> ${datastore_path}"

    pgctl_stop "${datastore_path}"

    rm -rf "${datastore_path}"
    cp -r "${datastore_clean_path}" "${datastore_path}"

    drop_os_page_cache

    export_env_from_ini "${config_path}" "${aker_config_override}"

    pgctl_start "${datastore_path}" "${psql_config_path}" "${pg_log_path}"
    pg_wait_ready "${host}" "${port}" "${user}"
}


restore_clean_snapshot_and_start_no_cache_drop() {
    #
    # Same as restore_clean_snapshot_and_start(), but does NOT drop OS page cache.
    # This is useful for workload generation or index build steps where OS cache state
    # is not part of the measured benchmark.
    #
    local config_path="$1"
    local purpose="$2"  # for log text only
    local aker_config_override="${3:-}"

    local datastore_path
    datastore_path="$(ini_get_value "${config_path}" "postgres" "datastore")"
    datastore_path="$(resolve_datastore_path "${datastore_path}")"
    if [[ -z "${datastore_path}" ]]; then
        # No PGDATA configured; caller likely manages Postgres externally.
        return 0
    fi

    local datastore_clean_path
    datastore_clean_path="$(ini_get_value "${config_path}" "postgres" "datastore_clean")"
    if [[ -z "${datastore_clean_path}" ]]; then
        datastore_clean_path="${datastore_path}-clean"
    fi

    datastore_clean_path="$(resolve_datastore_path "${datastore_clean_path}")"

    local psql_config_path
    psql_config_path="$(ini_get_value "${config_path}" "postgres" "psql_config")"

    if [[ -z "${psql_config_path}" && -f "${BENCH_ROOT}/configs/postgresql.conf" ]]; then
        psql_config_path="configs/postgresql.conf"
    fi

    psql_config_path="$(resolve_path_under_root "${psql_config_path}")"

    local pg_log_path
    pg_log_path="$(ini_get_value "${config_path}" "postgres" "pg_log")"
    if [[ -z "${pg_log_path}" ]]; then
        pg_log_path="output/postgres.log"
    fi
    pg_log_path="$(resolve_path_under_root "${pg_log_path}")"

    local host
    host="$(ini_get_value "${config_path}" "postgres" "host")"
    local port
    port="$(ini_get_value "${config_path}" "postgres" "port")"
    local user
    user="$(ini_get_value "${config_path}" "postgres" "user")"

    # Ensure a clean snapshot exists (optionally create it).
    ensure_clean_snapshot \
        "${config_path}" \
        "${datastore_path}" \
        "${datastore_clean_path}" \
        "${psql_config_path}" \
        "${pg_log_path}" \
        "${host}" \
        "${port}" \
        "${user}" \
        "${aker_config_override}"

    log_info "Restoring clean snapshot for ${purpose} (no cache drop): ${datastore_clean_path} -> ${datastore_path}"

    pgctl_stop "${datastore_path}"

    rm -rf "${datastore_path}"
    cp -r "${datastore_clean_path}" "${datastore_path}"

    export_env_from_ini "${config_path}" "${aker_config_override}"

    pgctl_start "${datastore_path}" "${psql_config_path}" "${pg_log_path}"
    pg_wait_ready "${host}" "${port}" "${user}"
}

capture_tmp_trace_list() {
    #
    # Capture current /tmp trace candidates into a sorted list file.
    #
    local out_path="$1"

    local tmp_file
    tmp_file="${out_path}.tmp"
    rm -f "${tmp_file}"

    # Use nullglob so non-matching globs expand to nothing.
    shopt -s nullglob

    local patterns=(
        /tmp/aker_trace_* 
        /tmp/topkache_trace_* 
        /tmp/aker_*trace* 
        /tmp/topkache_*trace*
    )

    local pattern
    for pattern in "${patterns[@]}"; do
        local p
        for p in ${pattern}; do
            if [[ -e "${p}" ]]; then
                printf "%s\n" "${p}" >> "${tmp_file}"
            fi
        done
    done

    shopt -u nullglob

    if [[ -f "${tmp_file}" ]]; then
        sort -u "${tmp_file}" > "${out_path}"
        rm -f "${tmp_file}"
    else
        : > "${out_path}"
    fi
}

collect_new_tmp_traces() {
    #
    # Compare /tmp trace candidates before/after and copy newly created ones to dest dirs.
    #
    # Args:
    #   $1: before_list_path
    #   $2: run_dest_dir (per-run)
    #   $3: merged_dest_dir (aggregated)
    #
    local before_list_path="$1"
    local run_dest_dir="$2"
    local merged_dest_dir="$3"

    mkdir -p "${run_dest_dir}"
    mkdir -p "${merged_dest_dir}"

    local after_list_path
    after_list_path="${before_list_path}.after"

    capture_tmp_trace_list "${after_list_path}"

    local new_list_path
    new_list_path="${before_list_path}.new"

    comm -13 "${before_list_path}" "${after_list_path}" > "${new_list_path}" || true

    while IFS= read -r path; do
        if [[ -z "${path}" ]]; then
            continue
        fi

        local base
        base="$(basename -- "${path}")"

        # Ensure unique names when copying.
        local run_target
        run_target="${run_dest_dir}/${base}"
        if [[ -e "${run_target}" ]]; then
            run_target="${run_dest_dir}/${base}_$(date +%s%N)"
        fi

        local merged_target
        merged_target="${merged_dest_dir}/${base}"
        if [[ -e "${merged_target}" ]]; then
            merged_target="${merged_dest_dir}/${base}_$(date +%s%N)"
        fi

        log_info "Collecting new /tmp trace: ${path} -> ${run_target}"
        cp -a "${path}" "${run_target}"
        cp -a "${path}" "${merged_target}"

    done < "${new_list_path}"

    rm -f "${after_list_path}" "${new_list_path}"
}
