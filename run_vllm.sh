#!/bin/bash

# Ensure that only the specified GPUs are available for the jobs
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Data folder path and result file path
data_folder="/PHShome/cs1839/capstone_data/"
results_df_path="${data_folder}results.csv"

# List of test data files
test_files=("mimic_iv_snippets_list.csv")

# Combinations for one_shot, cot, and five_shot
combinations=(
    "false false false"
    "true false false"
    "true true false"
    "false false true"
    "false true true"
)

# List of models and their paths
declare -A name_model_paths=( 
    ["Llama-3.1-8B"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.1-8B"
    ["Llama-3.1-8B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.1-8B-Instruct"
    ["Llama-3.1-70B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.1-70B-Instruct"
    ["Llama-3.2-1B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.2-1B-Instruct"
    ["Llama-3.2-3B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.2-3B-Instruct"
    ["MeLLaMA-13B"]="/netapp3/raw_data3/share/llm_public_host/test/physionet.org/files/me-llama/1.0.0/MeLLaMA-13B"
    ["MeLLaMA-13B-chat"]="/netapp3/raw_data3/share/llm_public_host/test/physionet.org/files/me-llama/1.0.0/MeLLaMA-13B-chat"
    ["Qwen2-72B-Instruct"]="/PHShome/jn180/llm_public_host/Qwen2-72B-Instruct"
    ["Qwen2-7B-Instruct"]="/PHShome/jn180/llm_public_host/Qwen2-7B-Instruct"
    ["Qwen2.5-14B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Qwen2.5-14B-Instruct"
    ["Qwen2.5-32B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Qwen2.5-32B-Instruct"
    ["Qwen2.5-72B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Qwen2.5-72B-Instruct"
    ["Mistral-7B-Instruct-v0.3"]="/netapp3/raw_data3/share/llm_public_host/Mistral-7B-Instruct-v0.3"
    ["Mistral-Nemo-Instruct-2407"]="/netapp3/raw_data3/share/llm_public_host/Mistral-Nemo-Instruct-2407"
    ["meditron-7b"]="/PHShome/jn180/llm_public_host/meditron-7b"
    ["meditron-70b"]="/PHShome/jn180/llm_public_host/meditron-70b"
)

# Extract the keys in the order of declaration
ordered_model_names=()
for model_name in "${!name_model_paths[@]}"; do
    ordered_model_names+=("$model_name")
done

# Iterate over each combination of test file and method
for test_df_name in "${test_files[@]}"; do
    # Set the dataset and prompt template key based on the test file
    if [[ "$test_df_name" == "medication_status_test.csv" ]]; then
        dataset_name="MIT"
        prompt_template_key="Other"
    elif [[ "$test_df_name" == "PPV_snippet_medications.csv" ]]; then
        dataset_name="Internal Data"
        prompt_template_key="Internal Data"
    elif [[ "$test_df_name" == "mimic_iv_snippets_list.csv" ]]; then
        dataset_name="MIMIC-IV"
        prompt_template_key="Other" 
    fi

    for combination in "${combinations[@]}"; do
        read -r one_shot cot five_shot <<< "$combination"
        echo "Running combination: one_shot=$one_shot, cot=$cot, five_shot=$five_shot"

        # Iterate over the ordered keys
        for model_name in "${ordered_model_names[@]}"; do
            model_path="${name_model_paths[$model_name]}"
            echo "Running model: $model_name with path: $model_path"
            echo "Using prompt template key: $prompt_template_key"

            # Run the Python script for each model and prompt template
            python vllm_inference.py \
                --model_path "$model_path" \
                --test_df_name "$test_df_name" \
                --prompt_template_key "$prompt_template_key" \
                --dataset_name "$dataset_name" \
                --data_folder "$data_folder" \
                --result_df_path "$results_df_path" \
                --one_shot "$one_shot" \
                --cot "$cot" \
                --five_shot "$five_shot" \
                --batch_size 150

            # Clear the CUDA cache to free up GPU memory
            echo "Clearing GPU memory after running model: $model_name"

            # Ensure garbage collection and add a short delay
            sleep 1
        done
    done
done

echo "All jobs completed."