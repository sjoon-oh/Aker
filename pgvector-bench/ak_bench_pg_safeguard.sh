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
# Docker support (current refactor):
# - PostgreSQL server is executed inside a container.
# - Host-side scripts still manage the workload pipeline and storage directories.
# - To enable Docker mode you MUST specify a Docker image name.
#   (Non-Docker execution is treated as legacy and requires explicit opt-in.)
#

BENCH_ROOT="$(pwd)"
PROJECT_ROOT="$(cd "${BENCH_ROOT}/.." && pwd)"

# Docker runtime state (set by prepare_docker_environment()).
# NOTE: Do not reuse the same names as user-facing env vars.
AK_BENCH_DOCKER_IMAGE_INTERNAL=""
AK_BENCH_DOCKER_CONTAINER_INTERNAL=""
AK_BENCH_DOCKER_NETWORK_INTERNAL="host"
AK_BENCH_DOCKER_OS_USER_INTERNAL="akerbench"
AK_BENCH_DOCKER_NUMA_NODE_INTERNAL="0"
AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT_INTERNAL="0"

declare -a AK_BENCH_DOCKER_ENV_ARGS=()

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

sanitize_container_component() {
    #
    # Convert an arbitrary string to a Docker container-name-safe token.
    #
    local raw="$1"
    printf "%s" "${raw}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9_.-]+/_/g'
}

docker_get_image() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_IMAGE:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_IMAGE}"
        return 0
    fi

    local ini_image
    ini_image="$(ini_get_value "${config_path}" "docker" "image" || true)"
    printf "%s" "${ini_image}"
}

docker_is_legacy_opt_in() {
    [[ "${AK_BENCH_ALLOW_LEGACY_NO_DOCKER:-0}" == "1" ]]
}

docker_require_image_or_legacy() {
    local config_path="$1"

    local image
    image="$(docker_get_image "${config_path}")"

    if [[ -n "${image}" ]]; then
        return 0
    fi

    # Guard against the old pattern: docker.enabled=1 without specifying an image.
    local enabled
    enabled="$(ini_get_value "${config_path}" "docker" "enabled" || true)"
    if [[ "${enabled}" == "1" ]]; then
        log_err "docker.enabled=1 is deprecated. You must specify docker.image (or set AK_BENCH_DOCKER_IMAGE)."
        exit 1
    fi

    if docker_is_legacy_opt_in; then
        log_warn "Docker image not specified; running in legacy (non-Docker) mode due to AK_BENCH_ALLOW_LEGACY_NO_DOCKER=1"
        return 0
    fi

    log_err "Docker mode is required. Specify a Docker image name via:"
    log_err "  - environment: export AK_BENCH_DOCKER_IMAGE=<image:tag>"
    log_err "  - or config INI: [docker] image = <image:tag>"
    log_err "To run legacy host Postgres, set AK_BENCH_ALLOW_LEGACY_NO_DOCKER=1 (not recommended for review)."
    exit 1
}

docker_get_network() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_NETWORK:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_NETWORK}"
        return 0
    fi

    local ini_network
    ini_network="$(ini_get_value "${config_path}" "docker" "network" || true)"
    if [[ -z "${ini_network}" ]]; then
        ini_network="host"
    fi
    printf "%s" "${ini_network}"
}

docker_get_numa_node() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_NUMA_NODE:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_NUMA_NODE}"
        return 0
    fi

    local ini_node
    ini_node="$(ini_get_value "${config_path}" "docker" "numa_node" || true)"
    if [[ -z "${ini_node}" ]]; then
        ini_node="0"
    fi

    printf "%s" "${ini_node}"
}

docker_get_remove_container_on_exit() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT}"
        return 0
    fi

    local ini_value
    ini_value="$(ini_get_value "${config_path}" "docker" "remove_container_on_exit" || true)"
    if [[ -n "${ini_value}" ]]; then
        printf "%s" "${ini_value}"
        return 0
    fi

    # Legacy alias: rm_on_exit means removing the *container*, not the image.
    ini_value="$(ini_get_value "${config_path}" "docker" "rm_on_exit" || true)"
    if [[ -n "${ini_value}" ]]; then
        printf "%s" "${ini_value}"
        return 0
    fi

    printf "0"
}

docker_get_container_name() {
    local config_path="$1"
    local image_tag="$2"
    local host_port="$3"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_NAME:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_CONTAINER_NAME}"
        return 0
    fi

    local ini_name
    ini_name="$(ini_get_value "${config_path}" "docker" "container_name" || true)"
    if [[ -n "${ini_name}" ]]; then
        printf "%s" "${ini_name}"
        return 0
    fi

    local token
    token="$(sanitize_container_component "${image_tag}")"
    printf "akerbench_%s_%s" "${token}" "${host_port}"
}

docker_is_enabled_for_config() {
    local config_path="$1"
    local image
    image="$(docker_get_image "${config_path}")"
    [[ -n "${image}" ]]
}

docker_ensure_container_running() {
    local config_path="$1"

    if ! docker_is_enabled_for_config "${config_path}"; then
        return 0
    fi

    require_cmd docker

    local image_tag
    image_tag="$(docker_get_image "${config_path}")"

    local host_port
    host_port="$(ini_get_value "${config_path}" "postgres" "port" || true)"
    if [[ -z "${host_port}" ]]; then
        host_port="5432"
    fi

    local container_name
    container_name="$(docker_get_container_name "${config_path}" "${image_tag}" "${host_port}")"

    local network_mode
    network_mode="$(docker_get_network "${config_path}")"

    local run_script
    run_script="${PROJECT_ROOT}/docker/scripts/run_container.sh"
    if [[ ! -x "${run_script}" ]]; then
        log_err "Docker run script not found or not executable: ${run_script}"
        exit 1
    fi

    "${run_script}" "${image_tag}" "${container_name}" "${network_mode}" "${host_port}"

    AK_BENCH_DOCKER_IMAGE_INTERNAL="${image_tag}"
    AK_BENCH_DOCKER_CONTAINER_INTERNAL="${container_name}"
    AK_BENCH_DOCKER_NETWORK_INTERNAL="${network_mode}"
    AK_BENCH_DOCKER_OS_USER_INTERNAL="${AK_BENCH_DOCKER_OS_USER:-akerbench}"
    AK_BENCH_DOCKER_NUMA_NODE_INTERNAL="$(docker_get_numa_node "${config_path}")"
    AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT_INTERNAL="$(docker_get_remove_container_on_exit "${config_path}")"
}

prepare_docker_environment() {
    #
    # Initialize Docker runtime state for a given benchmark config.
    # If Docker is required (default), this validates docker.image is set.
    #
    local config_path="$1"

    docker_require_image_or_legacy "${config_path}"

    if docker_is_enabled_for_config "${config_path}"; then
        docker_ensure_container_running "${config_path}"
    fi
}

docker_exec_raw() {
    #
    # Execute a shell command inside the benchmark container.
    #
    local inner_cmd="$1"

    docker exec -u "${AK_BENCH_DOCKER_OS_USER_INTERNAL}" \
        "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" \
        bash -lc "cd '${BENCH_ROOT}' && ${inner_cmd}"
}

docker_exec_raw_with_env() {
    #
    # Execute a shell command inside the benchmark container with environment variables
    # propagated from [env] section.
    #
    local inner_cmd="$1"

    docker exec -u "${AK_BENCH_DOCKER_OS_USER_INTERNAL}" \
        "${AK_BENCH_DOCKER_ENV_ARGS[@]}" \
        "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" \
        bash -lc "cd '${BENCH_ROOT}' && ${inner_cmd}"
}

export_env_from_ini() {
    #
    # Export environment variables under [env] section so PostgreSQL can read them via getenv.
    # Keys must be valid shell variable names.
    #
    # Args:
    #   $1: config_path
    #   $2: optional AKER_CONFIG_PATH override (wins over INI)
    #
    local config_path="$1"
    local aker_config_override="${2:-}"

    AK_BENCH_DOCKER_ENV_ARGS=()

    while IFS=$'\t' read -r key value; do
        if [[ -z "${key}" ]]; then
            continue
        fi
        export "${key}=${value}"
        log_info "Exported env: ${key}=${value}"

        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            AK_BENCH_DOCKER_ENV_ARGS+=( -e "${key}=${value}" )
        fi
    done < <(ini_list_section_kv "${config_path}" "env" || true)

    if [[ -n "${aker_config_override}" ]]; then
        local resolved
        resolved="$(resolve_path_under_root "${aker_config_override}")"
        export AKER_CONFIG_PATH="${resolved}"
        log_info "Exported env override: AKER_CONFIG_PATH=${resolved}"

        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            AK_BENCH_DOCKER_ENV_ARGS+=( -e "AKER_CONFIG_PATH=${resolved}" )
        fi
    fi
}

pg_initdb() {
    #
    # Initialize a new PGDATA directory.
    #
    # Args:
    #   $1: datastore_path
    #   $2: db_superuser
    #
    local datastore_path="$1"
    local db_superuser="$2"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        docker_exec_raw "initdb -D '${datastore_path}' -U '${db_superuser}'"
        return 0
    fi

    require_cmd initdb
    initdb -D "${datastore_path}" -U "${db_superuser}"
}

pgctl_stop() {
    local datastore_path="$1"

    if [[ ! -d "${datastore_path}" ]]; then
        log_warn "datastore path not found; skip pg_ctl stop: ${datastore_path}"
        return 0
    fi

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        docker_exec_raw "pg_ctl -D '${datastore_path}' -m fast stop >/dev/null 2>&1 || true"
        return 0
    fi

    require_cmd pg_ctl

    # Stop only the instance using this data directory.
    pg_ctl -D "${datastore_path}" -m fast stop >/dev/null 2>&1 || true
}

pgctl_start() {
    local datastore_path="$1"
    local psql_config_path="$2"
    local pg_log_path="$3"

    if [[ ! -d "${datastore_path}" ]]; then
        log_err "datastore path not found; cannot pg_ctl start: ${datastore_path}"
        exit 1
    fi

    mkdir -p "$(dirname -- "${pg_log_path}")"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        local numa_node="${AK_BENCH_DOCKER_NUMA_NODE_INTERNAL}"

        local extra_opts=""
        if [[ -n "${psql_config_path}" ]]; then
            extra_opts="${extra_opts} -o --config-file=${psql_config_path}"
        fi

        docker_exec_raw_with_env "\
            set -euo pipefail; \
            numa_prefix=(); \
            if [[ -n '${numa_node}' ]]; then numa_prefix=(numactl --cpunodebind='${numa_node}' --membind='${numa_node}'); fi; \
            \"\${numa_prefix[@]}\" pg_ctl -D '${datastore_path}' -l '${pg_log_path}' ${extra_opts} start >/dev/null\
        "
        return 0
    fi

    require_cmd pg_ctl

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

    prepare_docker_environment "${config_path}"

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

    prepare_docker_environment "${config_path}"

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

maybe_stop_postgres() {
    #
    # Stop PostgreSQL if we are managing PGDATA (postgres.datastore is configured).
    #
    # This helper is intentionally a no-op when postgres.datastore is empty, so that
    # callers can support externally managed PostgreSQL instances.
    #
    # Args:
    #   $1: config_path
    #
    local config_path="$1"

    if [[ -z "${config_path}" ]]; then
        log_warn "Config path is empty; skip maybe_stop_postgres"
        return 0
    fi

    local datastore_value
    datastore_value="$(ini_get_value "${config_path}" "postgres" "datastore")"
    datastore_value="$(resolve_datastore_path "${datastore_value}")"
    if [[ -z "${datastore_value}" ]]; then
        return 0
    fi

    pgctl_stop "${datastore_value}"
}

maybe_wait_for_aker_trace_export() {
    #
    # Aker trace export can be expensive. To avoid prematurely shutting down the server
    # (or container) before traces are written, optionally wait before stopping.
    #
    # Policy:
    # - If AK_BENCH_TRACE_EXPORT_WAIT_SEC is set, use it (0 disables).
    # - Otherwise, if AKER_CONFIG_PATH is set, default to 600 seconds.
    #
    local wait_sec="${AK_BENCH_TRACE_EXPORT_WAIT_SEC:-}"

    if [[ -z "${wait_sec}" ]]; then
        if [[ -n "${AKER_CONFIG_PATH:-}" ]]; then
            wait_sec="600"
        else
            wait_sec="0"
        fi
    fi

    if [[ "${wait_sec}" == "0" ]]; then
        return 0
    fi

    log_info "Waiting ${wait_sec} seconds for trace export before shutdown"
    sleep "${wait_sec}"
}

docker_cleanup_tmp_traces() {
    #
    # Remove trace artifacts under /tmp inside the container to keep it reusable.
    #
    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        return 0
    fi

    docker_exec_raw "\
        set -euo pipefail; \
        shopt -s nullglob; \
        rm -rf /tmp/aker_trace_* /tmp/topkache_trace_* /tmp/aker_*trace* /tmp/topkache_*trace* || true\
    "
}

maybe_shutdown_docker_container() {
    #
    # Stop the benchmark container if Docker mode is enabled.
    # By default we keep the container for reuse; removal is optional.
    #
    local config_path="$1"

    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        return 0
    fi

    require_cmd docker

    log_info "Stopping benchmark container (not removing image): ${AK_BENCH_DOCKER_CONTAINER_INTERNAL}"
    docker stop "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" >/dev/null 2>&1 || true

    if [[ "${AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT_INTERNAL}" == "1" ]]; then
        log_warn "Removing container due to remove_container_on_exit=1: ${AK_BENCH_DOCKER_CONTAINER_INTERNAL}"
        docker rm "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" >/dev/null 2>&1 || true
    fi
}

capture_tmp_trace_list() {
    #
    # Capture current /tmp trace candidates into a sorted list file.
    #
    local out_path="$1"

    local tmp_file
    tmp_file="${out_path}.tmp"
    rm -f "${tmp_file}"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        docker_exec_raw "\
            set -euo pipefail; \
            shopt -s nullglob; \
            for p in /tmp/aker_trace_* /tmp/topkache_trace_* /tmp/aker_*trace* /tmp/topkache_*trace*; do \
                if [[ -e \"\${p}\" ]]; then printf '%s\\n' \"\${p}\"; fi; \
            done | sort -u\
        " > "${out_path}" || true

        if [[ ! -f "${out_path}" ]]; then
            : > "${out_path}"
        fi
        return 0
    fi

    # Host mode (legacy)
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

        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            log_info "Collecting new container /tmp trace: ${path} -> ${run_target}"
            docker cp "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}:${path}" "${run_target}" >/dev/null 2>&1 || true
            docker cp "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}:${path}" "${merged_target}" >/dev/null 2>&1 || true
        else
            log_info "Collecting new /tmp trace: ${path} -> ${run_target}"
            cp -a "${path}" "${run_target}"
            cp -a "${path}" "${merged_target}"
        fi

    done < "${new_list_path}"

    rm -f "${after_list_path}" "${new_list_path}"
}
