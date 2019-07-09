#!/bin/bash

NAME=gaussSeidelGPU
CODE=$NAME.cu
SIZE=3
DEF_NAME=L

echo $CODE
echo "Compiling"
nvcc  -D$DEF_NAME=$SIZE $CODE -o $NAME
./$NAME
