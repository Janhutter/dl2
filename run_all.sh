#!/bin/bash


for dataset in cifar10 cifar100 tin200
do
    for model in source norm tent eta eata energy sar shot pl
    do
    # [1, 17, 36, 91, 511]
        for seed in 1 17 36 91 511
        do 
        # echo "Running test for $dataset $model $seed"
        # sbatch test.job $dataset $model $seed
        python main.py --cfg cfgs/$dataset/$model.yaml RNG_SEED $seed
        done
    done
done

