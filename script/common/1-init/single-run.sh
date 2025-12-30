#!/bin/bash

# Check if the current root working directory is set.

export PATH=/usr/local/pgsql/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/pgsql/lib:$LD_LIBRARY_PATH
export MANPATH=/usr/local/pgsql/share/man:$MANPATH

export CPLUS_INCLUDE_PATH=/usr/local/pgsql/include
export C_INCLUDE_PATH=/usr/local/pgsql/include

PGASSWORD=passwd psql -h localhost -p 5432 -U postgres -f $1

