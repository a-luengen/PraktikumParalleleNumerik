#!/bin/bash

NAME=GaussSeidelVerfahren
CODE=$NAME.c
SIZE=3
DEF_NAME1=L
CC=gcc
FLAGS=-std=c99

echo $CODE
echo "Compiling"
$CC $FLAGS -D$DEF_NAME1=$SIZE $CODE -o $NAME
./$NAME
