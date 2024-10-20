#!/bin/bash

# Ensure that only the specified GPUs are available for the jobs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Data folder path and result file path
data_folder="/PHShome/cs1839/capstone_data/"
results_df_path="${data_folder}results.csv"
test_df_name="PPV_snippet_medications.csv"  # Adjust this depending on your actual test file
one_shot=false
cot=true


# Load the appropriate test dataframe
input_df="${data_folder}${test_df_name}"

# List of models and their paths
declare -A name_model_paths=(
    ["Llama-3.1-70B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.1-70B-Instruct"
    ["Qwen2-72B-Instruct"]="/PHShome/jn180/llm_public_host/Qwen2-72B-Instruct"
    ["Qwen2.5-32B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Qwen2.5-32B-Instruct"
    ["Qwen2.5-72B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Qwen2.5-72B-Instruct"
    ["meditron-70b"]="/PHShome/jn180/llm_public_host/meditron-70b"
    ["Mistral-7B-Instruct-v0.3"]="/netapp3/raw_data3/share/llm_public_host/Mistral-7B-Instruct-v0.3"
    ["Mistral-Nemo-Instruct-2407"]="/netapp3/raw_data3/share/llm_public_host/Mistral-Nemo-Instruct-2407"

    ["Llama-3.1-8B"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.1-8B"
    ["Llama-3.1-8B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.1-8B-Instruct"
    ["Llama-3.2-1B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.2-1B-Instruct"
    ["Llama-3.2-3B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Llama-3.2-3B-Instruct"
    ["Qwen2-7B-Instruct"]="/PHShome/jn180/llm_public_host/Qwen2-7B-Instruct"
    ["Qwen2.5-14B-Instruct"]="/netapp3/raw_data3/share/llm_public_host/Qwen2.5-14B-Instruct"
    ["meditron-7b"]="/PHShome/jn180/llm_public_host/meditron-7b"
)

# Define prompt template keys based on dataset
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

# Iterate over each model and run them one after another
for model_name in "${!name_model_paths[@]}"; do
    model_path="${name_model_paths[$model_name]}"

    echo "Using prompt template key: ${prompt_template_key}"

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
        --batch_size 200 \
        --max_token_output 200 \

    # Clear the CUDA cache to free up GPU memory
    echo "Clearing GPU memory after running model: ${model_name}"
    python -c "import torch; torch.cuda.empty_cache()"

    # Ensure garbage collection and add a short delay
    sleep 2
    python -c "import gc; gc.collect()"


done


echo "All jobs completed."