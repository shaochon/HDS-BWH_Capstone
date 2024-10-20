import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import gc
import os
import json
from huggingface_hub import login
import torch
import argparse
from ast import literal_eval

# 1. Function to initialize model and tokenizer
def initialize_model(model_path, device=0, use_fp16=True):
    """
    Initializes the model and tokenizer for text generation.
    
    Parameters:
    ----------
    model_path : str
        Path of the model to be loaded.
    device : int
        Device to use, 0 for GPU and -1 for CPU.
    use_fp16 : bool
        Whether to use FP16 for inference.
    
    Returns:
    -------
    generator : pipeline
        A HuggingFace pipeline ready for text generation.
    """
    # Load tokenizer and set padding side to left
    tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos_token (common for autoregressive models)
    tokenizer.padding_side = "left"  # Set padding to left for autoregressive models

    # Initialize the pipeline for text generation
    generator = pipeline(
        task="text-generation",
        model=model_path,
        tokenizer=tokenizer,  # Pass the tokenizer with left padding settings
        device=device,  # '0' for GPU, '-1' for CPU
        model_kwargs={"torch_dtype": torch.float16} if use_fp16 else {}
    )
    return generator


# 2. Function to generate batch responses using the model
def generate_responses(input_df, batch_size, generator, prompt_template, max_token_output=80, use_sampling=True):
    """
    Generate text responses in batches using the generator.
    
    Parameters:
    ----------
    input_df : list of str
        List of input texts to run inference on.
    batch_size : int
        Size of each batch for inference.
    generator : pipeline
        HuggingFace pipeline initialized for text generation.
    prompt_template : str
        The template for the prompt to be used.
    max_token_output : int
        Maximum number of tokens to generate.
    use_sampling : bool
        Whether to use sampling or greedy decoding.
    
    Returns:
    -------
    response_list : list of str
        List of generated responses.
    """
    sub_df = input_df['snippet'].values.tolist()

    response_list = []
    num_step = len(sub_df) // batch_size + (1 if len(sub_df) % batch_size != 0 else 0)
    temperature = 0.9 if use_sampling else 1
    top_p = 0.9 if use_sampling else 1
    top_k = 20 if use_sampling else None

    for i in tqdm(range(num_step)):
        input_texts = sub_df[i*batch_size:(i+1)*batch_size]
        input_texts = [prompt_template.format(text) for text in input_texts]

        responses = generator(
            input_texts,
            max_new_tokens=max_token_output,  # Ensure this is used for token generation
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
            do_sample=use_sampling,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k)

        # Process the output
        for response in responses:
            for generated in response:
                # Extract relevant part of the response and append to list
                response_list.append(generated['generated_text'].split("\n\nOutput:")[-1].strip())
    
    return response_list

# 3. Function to process the LLM output
def process_output(input_df, response_list, dataset_name):
    """
    Processes a list of LLM responses to extract medication information and adds it to the input DataFrame.

    This function takes an input DataFrame (`input_df`) and a list of responses (`response_list`),
    where each response contains categorized medication data. The function extracts three categories
    of medications (active, discontinued, and neither), formats them into lists, and creates a new
    DataFrame with three columns:
    
    - `active_medications`: Medications that the patient is currently taking.
    - `discontinued_medications`: Medications that the patient has taken but has since discontinued.
    - `neither_medications`: Medications that are mentioned but are neither currently taken nor discontinued.
      (Excluded for "Internal Data" dataset).
    
    The new DataFrame with these three columns is concatenated with the `input_df` and returned.

    Parameters:
    ----------
    input_df : pd.DataFrame
        The original input DataFrame, which will be concatenated with the extracted medication data.
    
    response_list : list of str
        A list of strings containing the LLM responses. Each response includes a categorized list of medications.
    
    dataset_name : str
        The name of the dataset. If "Internal Data", `neither_medications` will be excluded.
    
    Returns:
    -------
    pd.DataFrame
        A new DataFrame that concatenates the `input_df` with the extracted medication data.
    """
    # check if input_df.active_medications[0] is a list, if not, apply eval and lower to active_medications, discontinued_medications, neither_medications
    if not isinstance(input_df.active_medications[0], list):
        input_df.loc[:, 'active_medications'] = input_df['active_medications'].apply(lambda x: literal_eval(x)).apply(lambda x: [med.lower() for med in x])
        input_df.loc[:, 'discontinued_medications'] = input_df['discontinued_medications'].apply(lambda x: literal_eval(x)).apply(lambda x: [med.lower() for med in x])
        if dataset_name != "Internal Data":
            input_df.loc[:, 'neither_medications'] = input_df['neither_medications'].apply(lambda x: literal_eval(x)).apply(lambda x: [med.lower() for med in x])

    # Initialize lists to store the medications for each category
    active_medications_list = []
    discontinued_medications_list = []
    neither_medications_list = []

    for response in response_list:
        # Extract the active, discontinued, and neither medications using regular expressions
        active_medications = re.search(r'Current Medications \(Active\):\s*(.*?)(?:\n- Discontinued Medications:)', response, re.IGNORECASE)
        
        # Only extract neither_medications if the dataset contains that information
        if dataset_name != "Internal Data":
            discontinued_medications = re.search(r'Discontinued Medications:\s*(.*?)(?:\n- Other Mentioned Medications)', response, re.IGNORECASE)
            neither_medications = re.search(r"Other Mentioned Medications \(neither active nor discontinued\): (.*?)(?:\s*END|\n|$)", response, re.IGNORECASE | re.DOTALL)
        else:
            discontinued_medications = re.search(r'Discontinued Medications:\s*(.*?)(?:\s*END|\n|$)', response, re.IGNORECASE)
            neither_medications = None

        # Convert to lists and handle None cases
        active_medications = active_medications.group(1).split(', ') if active_medications and active_medications.group(1).strip().lower() not in ['none',''] else []
        discontinued_medications = discontinued_medications.group(1).split(', ') if discontinued_medications and discontinued_medications.group(1).strip().lower() not in ['none',''] else []
        neither_medications = neither_medications.group(1).split(', ') if neither_medications and neither_medications.group(1).strip().lower() not in ['none',''] else []

        # Append each category list to their respective main lists
        active_medications_list.append(active_medications)
        discontinued_medications_list.append(discontinued_medications)
        if dataset_name != "Internal Data":
            neither_medications_list.append(neither_medications)

    # Create a new DataFrame from the lists
    output_df = pd.DataFrame({
        'model_response': response_list,
        'active_medications_pred': active_medications_list,
        'discontinued_medications_pred': discontinued_medications_list
    })
    
    # Include neither_medications only if the dataset is not "Internal Data"
    if dataset_name != "Internal Data":
        output_df['neither_medications_pred'] = neither_medications_list

    # Concatenate the input_df with the output_df
    result_df = pd.concat([input_df, output_df], axis=1)

    # Convert all predictions to lowercase
    result_df.loc[:, 'active_medications_pred'] = result_df['active_medications_pred'].apply(lambda x: [med.lower() for med in x])
    result_df.loc[:, 'discontinued_medications_pred'] = result_df['discontinued_medications_pred'].apply(lambda x: [med.lower() for med in x])
    if dataset_name != "Internal Data":
        result_df.loc[:, 'neither_medications_pred'] = result_df['neither_medications_pred'].apply(lambda x: [med.lower() for med in x])

    return result_df

# 4. Function to calculate metrics (Precision, Recall, F1, Accuracy)
def calculate_metrics_by_dataset(df, dataset_name):
    # Combine true and predicted sets
    if dataset_name == 'Internal Data':
        df['true_set'] = df[['active_medications', 'discontinued_medications']].apply(lambda x: [med for meds in x for med in meds], axis=1)
        df['pred_set'] = df[['active_medications_pred', 'discontinued_medications_pred']].apply(lambda x: [med for meds in x for med in meds], axis=1)
    else:
        df['true_set'] = df[['active_medications', 'discontinued_medications', 'neither_medications']].apply(lambda x: [med for meds in x for med in meds], axis=1)
        df['pred_set'] = df[['active_medications_pred', 'discontinued_medications_pred', 'neither_medications_pred']].apply(lambda x: [med for meds in x for med in meds], axis=1)

    # Calculate intersection of true and predicted sets
    df['intersection'] = df.apply(lambda row: set(row['pred_set']).intersection(set(row['true_set'])), axis=1)

    # Calculate true_count, pred_count, intersection_count
    df['true_count'] = df['true_set'].apply(lambda x: len(x))
    df['pred_count'] = df['pred_set'].apply(lambda x: len(x))
    df['intersection_count'] = df['intersection'].apply(lambda x: len(x))

    # Summing over the columns to calculate a micro precision and recall
    true_count = df['true_count'].sum()
    pred_count = df['pred_count'].sum()
    intersection_count = df['intersection_count'].sum()

    # Calculate extraction precision, recall, and f1
    extraction_precision = intersection_count / pred_count if pred_count > 0 else 0
    extraction_recall = intersection_count / true_count if true_count > 0 else 0
    extraction_f1 = 2 * extraction_precision * extraction_recall / (extraction_precision + extraction_recall) if (extraction_precision + extraction_recall) > 0 else 0

    # Task 2 metric calculation function
    def calculate_task2_metrics(df, med_class):
        # Define columns dynamically based on the class
        true_col = f'{med_class}_medications'
        pred_col = f'{med_class}_medications_pred'
        pred_task2_col = f'{med_class}_medications_pred_task2'
        task2_pred_count_col = f'task2_{med_class}_pred_count'
        task2_true_count_col = f'task2_{med_class}_true_count'
        task2_intersection_count_col = f'task2_{med_class}_intersection_count'
        
        # Calculate intersection for task 2
        df[pred_task2_col] = df.apply(lambda row: [med for med in row[pred_col] if med in row['intersection']], axis=1)
        
        # Calculate counts for task 2 precision and recall
        df[task2_pred_count_col] = df[pred_task2_col].apply(lambda x: len(x))
        df[task2_true_count_col] = df[true_col].apply(lambda x: len(x))
        df[task2_intersection_count_col] = df.apply(lambda row: len(set(row[true_col]).intersection(set(row[pred_task2_col]))), axis=1)
        
        # Calculate precision, recall, and F1 for the given class
        precision = df[task2_intersection_count_col].sum() / df[task2_pred_count_col].sum() if df[task2_pred_count_col].sum() > 0 else 0
        recall = df[task2_intersection_count_col].sum() / df[task2_true_count_col].sum() if df['true_count'].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return df, precision, recall, f1

    # Apply task2 metrics calculation for each medication class
    df, active_precision, active_recall, active_f1 = calculate_task2_metrics(df, 'active')
    df, discontinued_precision, discontinued_recall, discontinued_f1 = calculate_task2_metrics(df, 'discontinued')

    if dataset_name != 'Internal Data':
        df, neither_precision, neither_recall, neither_f1 = calculate_task2_metrics(df, 'neither')
    else:
        neither_precision, neither_recall, neither_f1 = 0, 0, 0

    # Add a column to sum the correct predictions for the 3 classes
    df['correct_pred_count'] = df['task2_active_intersection_count'] + df['task2_discontinued_intersection_count'] + (df['task2_neither_intersection_count'] if dataset_name != 'Internal Data' else 0)

    # Calculate the macro metrics
    conditional_accuracy = df['correct_pred_count'].sum() / df['pred_count'].sum() if df['true_count'].sum() > 0 else 0   
    conditional_macro_f1 = (active_f1 + discontinued_f1 + neither_f1) / (2 if dataset_name == 'Internal Data' else 3)
    conditional_macro_precision = (active_precision + discontinued_precision + neither_precision) / (2 if dataset_name == 'Internal Data' else 3)
    conditional_macro_recall = (active_recall + discontinued_recall + neither_recall) / (2 if dataset_name == 'Internal Data' else 3)

    return extraction_precision, extraction_recall, extraction_f1, conditional_accuracy, conditional_macro_f1, conditional_macro_precision, conditional_macro_recall

# 5. Main function to tie everything together
def run_pipeline(model_path, input_df, prompt_template, dataset_name, batch_size=16, max_token_output=80, use_sampling=False):
    """
    Main function to run the text generation pipeline and compute metrics.
    
    Parameters:
    ----------
    model_path : str
        The path of the model to be used.
    input_df : pd.DataFrame
        The data to be inferred.
    prompt_template : str
        Template for constructing the prompts.
    dataset_name : str
        Name of the dataset. If "Internal Data", `neither_medications` will be excluded.
    batch_size : int
        Number of examples per batch.
    max_token_output : int
        Maximum number of tokens to generate.
    use_sampling : bool
        Whether to use sampling (or greedy decoding).
    
    Returns:
    -------
    result_df : pd.DataFrame
        DataFrame with the processed outputs and calculated metrics.
    """
    # Initialize the model
    generator = initialize_model(model_path, device=0)

    # Generate responses
    response_list = generate_responses(input_df, batch_size, generator, prompt_template, max_token_output, use_sampling)

    # Process the responses to categorize medications
    df_w_classifications = process_output(input_df, response_list, dataset_name)

    # Calculate row-level metrics
    extraction_precision, extraction_recall, extraction_f1, conditional_accuracy, conditional_macro_f1, conditional_macro_precision, conditional_macro_recall = calculate_metrics_by_dataset(df_w_classifications, dataset_name)

    # Return the final DataFrame with metrics
    return df_w_classifications, extraction_precision, extraction_recall, extraction_f1, conditional_accuracy, conditional_macro_f1, conditional_macro_precision, conditional_macro_recall

# 6. Function to benchmark the model
def benchmark_model(name_dataset, model_path, prompt_template, input_df, data_folder, result_df_path, use_sampling=False, batch_size=16, max_token_output=80):
    """
    Function to run the entire pipeline and compute average row-wise metrics for a specific model and dataset.

    Parameters:
    ----------
    name_dataset : str
        The name of the dataset being processed (e.g., "Internal Data" or "MIT").
    model_path : str
        The path of the model to be used.
    prompt_template : str
        Template for constructing the prompts.
    input_df : pd.DataFrame
        Data to run inference on.
    data_folder : str
        The folder where input data and results are stored.
    result_df_path : str
        Path to the CSV file where results will be stored.
    use_sampling : bool
        Whether to use sampling (or greedy decoding).
    batch_size : int
        Number of examples per batch.
    max_token_output : int
        Maximum number of tokens to generate.
    """
    model_name = model_path.split('/')[-1]
    # Run the pipeline
    df_w_classifications, extraction_precision, \
    extraction_recall, extraction_f1, conditional_accuracy, \
    conditional_macro_f1, conditional_macro_precision, \
        conditional_macro_recall = run_pipeline(model_path=model_path, 
                                    input_df=input_df, 
                                    prompt_template=prompt_template, 
                                    dataset_name=name_dataset,
                                    use_sampling=use_sampling,
                                    batch_size=batch_size, 
                                    max_token_output=max_token_output)
    
    df_w_classifications.to_csv(data_folder+f'base_pred_data/{name_dataset}_{model_name}.csv', index=False)

    result_df = pd.read_csv(data_folder+'results.csv')

    # Define your result row
    new_row = {
        'Dataset': name_dataset,
        'Model': model_name,
        'Prompt': prompt_template,
        'extraction_precision': extraction_precision,
        'extraction_recall': extraction_recall,
        'extraction_f1': extraction_f1,
        'conditional_accuracy': conditional_accuracy,
        'conditional_macro_f1': conditional_macro_f1,
        'conditional_macro_precision': conditional_macro_precision,
        'conditional_macro_recall': conditional_macro_recall
    }

    result_df = result_df._append(new_row, ignore_index=True).round(3)
    result_df.to_csv(result_df_path, index=False)


def clear_cuda_memory():
    """Clear the CUDA cache and run garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()

# Function to define and parse command-line arguments
def parse_arguments():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description='Run inference with LLM models.')
    
    # Add arguments for CUDA device, test file name, and prompt template
    parser.add_argument('--cuda_device', type=str, default="1", help="Specify which CUDA device to use.")
    parser.add_argument('--test_df_name', type=str, required=True, help="Specify the test CSV filename.")
    parser.add_argument('--prompt_template_file', type=str, required=True, help="Specify the prompt template file for the model.")
    
    return parser.parse_args()

if "__main__" == __name__:
    # Parse command-line arguments
    args = parse_arguments()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    # Set the prompt template
    # Read the prompt template from a json file
    with open(args.prompt_template_file, 'r') as file:
        prompt_template = json.load(file)

    # Set the inference data file
    input_df_name = args.test_df_name

    # Read the JSON config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Get the token from the JSON file
    hg_token = config['HuggingFace']['token']
    # Login using the token
    login(token=hg_token)

    # Data folder
    data_folder = "/PHShome/cs1839/capstone_data/"
    # Results table path
    results_df_path = data_folder + "results.csv"

    # Load the inference data
    test_df = pd.read_csv(data_folder + input_df_name)
    if input_df_name == 'medication_status_test.csv':
        dataset_name = "MIT"
        prompt_template = prompt_template['Other']
    elif input_df_name == 'PPV_snippet_medications.csv':
        dataset_name = "Internal Data"
        prompt_template = prompt_template['Internal Data']
    elif input_df_name == 'mimic_iv_snippets_list.csv':
        dataset_name = "MIMIC-IV"
        prompt_template = prompt_template['Other']

    # Define the model paths
    name_model_paths ={   
        # "Bio_ClinicalBERT": "/PHShome/jn180/llm_public_host/Bio_ClinicalBERT",
        "Llama-3.1-8B": "/netapp3/raw_data3/share/llm_public_host/Llama-3.1-8B",
        "Llama-3.1-8B-Instruct": "/netapp3/raw_data3/share/llm_public_host/Llama-3.1-8B-Instruct",
        "Llama-3.2-1B-Instruct": "/netapp3/raw_data3/share/llm_public_host/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct": "/netapp3/raw_data3/share/llm_public_host/Llama-3.2-3B-Instruct",
        "Qwen2-7B-Instruct": "/PHShome/jn180/llm_public_host/Qwen2-7B-Instruct",
        "Qwen2.5-14B-Instruct": "/netapp3/raw_data3/share/llm_public_host/Qwen2.5-14B-Instruct",
        "meditron-7b": "/PHShome/jn180/llm_public_host/meditron-7b"
    }
    print(prompt_template)
    # Run the benchmark for each model
    for model_name, model_path in name_model_paths.items():
        clear_cuda_memory()
        benchmark_model(name_dataset=dataset_name,
                        model_path=model_path,
                        prompt_template=prompt_template,
                        input_df=test_df,
                        data_folder=data_folder,
                        result_df_path=results_df_path,
                        use_sampling=False,
                        batch_size=200,
                        max_token_output=80)


