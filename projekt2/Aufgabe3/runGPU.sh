#!/bin/bash

NAME=gaussSeidelGPU
CODE=$NAME.cu
SIZE=3
DEF_NAME1=L
DEF_NAME2=PRINT

echo "Compiling"
echo "#########"
#nvcc -D$DEF_NAME1=$SIZE -D$DEF_NAME2 $CODE -o $NAME
nvcc -D$DEF_NAME1=$SIZE $CODE -o $NAME

start=`date +%s%N`
./$NAME
end=`date +%s%N`
echo 1. Execution time was `expr $end - $start` nanoseconds.
start=`date +%s%N`
./$NAME
end=`date +%s%N`
echo 2. Execution time was `expr $end - $start` nanoseconds.
start=`date +%s%N`
./$NAME
end=`date +%s%N`
echo 3. Execution time was `expr $end - $start` nanoseconds.