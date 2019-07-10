#!/bin/bash

NAME=gaussSeidelGPU
CODE=$NAME.cu
SIZE=2
DEF_NAME1=L
DEF_NAME2=PRINT

echo $CODE
echo "Compiling"
nvcc -D$DEF_NAME1=$SIZE -D$DEF_NAME2 $CODE -o $NAME
./$NAME
