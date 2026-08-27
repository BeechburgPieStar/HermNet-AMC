#!/bin/bash
# Sweep over --version A..F with fixed best hyperparameters

modelname="HermNet"

# Fixed best hyperparameters from previous search
d=16
f=8
g=3
r=2

# Versions to sweep
versions=('full')

mkdir -p logs

for v in "${versions[@]}"; do
    tag="d${d}_f${f}_g${g}_r${r}_v${v}"
    echo "==============================================="
    echo "Running version=${v}  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "==============================================="

    python3 main.py --cuda_device 0 --mode train_test --model_name $modelname \
        --d_model $d --fused_dim $f --gmlp_layers $g --reduction $r \
        --dataset_name CD2025 --num_cls 10 \
        --dataset_indexes_for_train L2 L3 L4 --dataset_indexes_for_test L1 \
        > logs/${modelname}_L1_${tag}.log 2>&1 &

    python3 main.py --cuda_device 1 --mode train_test --model_name $modelname \
        --d_model $d --fused_dim $f --gmlp_layers $g --reduction $r \
        --version $v \
        --dataset_name CD2025 --num_cls 10 \
        --dataset_indexes_for_train L1 L3 L4 --dataset_indexes_for_test L2 \
        > logs/${modelname}_L2_${tag}.log 2>&1 &

    python3 main.py --cuda_device 2 --mode train_test --model_name $modelname \
        --d_model $d --fused_dim $f --gmlp_layers $g --reduction $r \
        --version $v \
        --dataset_name CD2025 --num_cls 10 \
        --dataset_indexes_for_train L1 L2 L4 --dataset_indexes_for_test L3 \
        > logs/${modelname}_L3_${tag}.log 2>&1 &

    python3 main.py --cuda_device 3 --mode train_test --model_name $modelname \
        --d_model $d --fused_dim $f --gmlp_layers $g --reduction $r \
        --version $v \
        --dataset_name CD2025 --num_cls 10 \
        --dataset_indexes_for_train L1 L2 L3 --dataset_indexes_for_test L4 \
        > logs/${modelname}_L4_${tag}.log 2>&1 &

    wait
    echo "Version ${v} done."
done

echo "All versions complete. Now run: python3 collect_results.py"