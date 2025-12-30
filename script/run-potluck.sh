#!/bin/bash
#
#

cd ..
source ./script/activate.sh

cd script

cd pgvector-01perc
./potluck.sh
cd ..

cd pgvector-02perc
./potluck.sh
cd ..

cd pgvector-03perc
./potluck.sh
cd ..

cd pgvector-04perc
./potluck.sh
cd ..

cd pgvector-05perc
./potluck.sh
cd ..

cd pgvector-10perc
./potluck.sh
cd ..

cd pgvector-20perc
./potluck.sh
cd ..
