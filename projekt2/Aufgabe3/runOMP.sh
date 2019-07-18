#!/bin/bash

NAME=gaussSeidelOMP
CODE=$NAME.c
SIZE=3
DEF_NAME1=L
CC=gcc
FLAGS="-std=c99 -fopenmp"
echo "Compiling"
echo "#########"
$CC $FLAGS -D$DEF_NAME1=$SIZE $CODE -o $NAME -lm
./$NAME

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