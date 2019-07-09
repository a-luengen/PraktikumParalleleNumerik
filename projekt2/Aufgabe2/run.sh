#!/bin/bash

NAME=main2
CODE=$NAME.cu
SIZE=34
DEF_NAME=ARRAY_SIZE

echo $CODE
echo "Compiling"
nvcc  -D$DEF_NAME=$SIZE $CODE -o $NAME
./$NAME
