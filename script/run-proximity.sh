#!/bin/bash
#
#

cd ..
source ./script/activate.sh

cd script

cd pgvector-01perc
./proximity.sh
cd ..

cd pgvector-02perc
./proximity.sh
cd ..

cd pgvector-03perc
./proximity.sh
cd ..

cd pgvector-04perc
./proximity.sh
cd ..

cd pgvector-05perc
./proximity.sh
cd ..

cd pgvector-10perc
./proximity.sh
cd ..

cd pgvector-20perc
./proximity.sh
cd ..
