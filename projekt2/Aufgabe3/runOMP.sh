#!/bin/bash

NAME=gaussSeidelOMP
CODE=$NAME.c
SIZE=3
DEF_NAME1=L
CC=gcc
FLAGS="-std=c99 -fopenmp"
echo $FLAGS
echo $CODE
echo "Compiling"
echo "#######"
$CC $FLAGS -D$DEF_NAME1=$SIZE $CODE -o $NAME -lm
./$NAME
