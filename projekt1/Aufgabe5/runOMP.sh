#!/bin/bash

THREAD_COUNT=8
NAME=partielleDifferentierung
CODE=$NAME.c
SIZE=5
DEF_NAME1=L
CC=gcc
FLAGS="-std=c99 -fopenmp"
echo "Compiling"
echo "#########"
$CC $FLAGS -D$DEF_NAME1=$SIZE $CODE -o $NAME -lm

export OMP_NUM_THREADS=$THREAD_COUNT
echo "Running with $OMP_NUM_THREADS threads."
./$NAME

./$NAME | grep "Time used"
./$NAME | grep "Time used"
./$NAME | grep "Time used"

exit 0
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
