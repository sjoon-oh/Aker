#!/bin/bash
#
#

cd ..
source ./script/activate.sh

cd script

cd pgvector-01perc
./topkache.sh
cd ..

cd pgvector-02perc
./topkache.sh
cd ..

cd pgvector-03perc
./topkache.sh
cd ..

cd pgvector-04perc
./topkache.sh
cd ..

cd pgvector-05perc
./topkache.sh
cd ..

cd pgvector-10perc
./topkache.sh
cd ..

cd pgvector-20perc
./topkache.sh
cd ..
