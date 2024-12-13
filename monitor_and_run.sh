#!/bin/bash

# Specify the GPUs to monitor
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_TO_CHECK="0,1,2,3"

# The interval in seconds to check GPU status
CHECK_INTERVAL=10

# Command to check GPU usage for specific GPUs
GPU_USAGE_CMD="nvidia-smi --id=${GPUS_TO_CHECK} --query-compute-apps=pid --format=csv,noheader"

# Command to start your job
YOUR_JOB_CMD="./run_vllm.sh"

echo "Monitoring GPUs ${GPUS_TO_CHECK}. Your job will start when GPUs are free..."

while true; do
    # Get the list of process IDs using the specified GPUs
    GPU_PROCESSES=$(eval $GPU_USAGE_CMD)

    if [ -z "$GPU_PROCESSES" ]; then
        echo "GPUs ${GPUS_TO_CHECK} are free. Starting your job..."
        eval $YOUR_JOB_CMD
        break
    else
        echo "GPUs ${GPUS_TO_CHECK} are busy. Checking again in $CHECK_INTERVAL seconds..."
        sleep $CHECK_INTERVAL
    fi
done