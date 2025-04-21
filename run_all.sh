#!/bin/bash
# dataset=tin200
dataset=cifar10
# dataset=cifar10

# model=sar
# source norm tent eta eata energy sar shot pl

for model in pl
do
# [1, 17, 36, 91, 511]
for seed in 1 17 36 91 511
do 
# echo "Running test for $dataset $model $seed"
sbatch test.job $dataset $model $seed
done
done

