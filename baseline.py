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
                response_list.append(generated['generated_text'].split("\n\nOutput:")[1].split("END")[0])
    
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
def calculate_row_metrics(df, dataset_name):
    """
    Calculate row-wise metrics, excluding `neither_medications` for "Internal Data".
    
    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing true and predicted medication categories.
    
    dataset_name : str
        Name of the dataset being processed. For "Internal Data", `neither_medications` will be excluded.
    
    Returns:
    -------
    pd.DataFrame
        The DataFrame updated with calculated metrics.
    """
    # Initialize columns to store row-wise metrics
    df.loc[:, 'extraction_precision'] = np.nan
    df.loc[:, 'extraction_recall'] = np.nan
    df.loc[:, 'conditional_accuracy'] = np.nan
    df.loc[:, 'conditional_macro_f1'] = np.nan
    df.loc[:, 'conditional_macro_precision'] = np.nan
    df.loc[:, 'conditional_macro_recall'] = np.nan

    # Helper function to compute F1, precision, and recall
    def compute_conditional_metrics(true_set, pred_set):
        tp = len(true_set.intersection(pred_set))
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0
        recall = tp / len(true_set) if len(true_set) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    for index, row in df.iterrows():
        # Combine predictions and true sets based on dataset
        if dataset_name == "Internal Data":
            pred_set = set(row['active_medications_pred'] + row['discontinued_medications_pred'])
            true_set = set(row['active_medications'] + row['discontinued_medications'])
        else:
            pred_set = set(row['active_medications_pred'] + row['discontinued_medications_pred'] + row['neither_medications_pred'])
            true_set = set(row['active_medications'] + row['discontinued_medications'] + row['neither_medications'])

        intersection = pred_set.intersection(true_set)

        # Compute precision and recall for extraction
        precision = len(intersection) / len(pred_set) if len(pred_set) != 0 else 0
        recall = len(intersection) / len(true_set) if len(true_set) != 0 else 0

        df.loc[index, 'extraction_precision'] = precision
        df.loc[index, 'extraction_recall'] = recall

        # Compute conditional metrics based on true and predicted sets
        correctly_extracted_active = set(row['active_medications']).intersection(intersection)
        correctly_extracted_active_pred = set(row['active_medications_pred']).intersection(intersection)
        # get the intersection of correctly extracted active medications and active medications predicted count
        active_acc_count = correctly_extracted_active.intersection(correctly_extracted_active_pred)

        correctly_extracted_discontinued = set(row['discontinued_medications']).intersection(intersection)
        correctly_extracted_discontinued_pred = set(row['discontinued_medications_pred']).intersection(intersection)
        discontinued_acc_count = correctly_extracted_discontinued_pred.intersection(correctly_extracted_discontinued)
        
        # Keep your original logic for neither_medications
        if dataset_name != "Internal Data":
            correctly_extracted_neither = set(row['neither_medications']).intersection(intersection)
            correctly_extracted_neither_pred = set(row['neither_medications_pred']).intersection(intersection)
            neither_acc_count = correctly_extracted_neither.intersection(correctly_extracted_neither_pred)
        else:
            correctly_extracted_neither = set()

        active_f1, active_precision, active_recall = compute_conditional_metrics(correctly_extracted_active, correctly_extracted_active_pred)
        discontinued_f1, discontinued_precision, discontinued_recall = compute_conditional_metrics(correctly_extracted_discontinued, correctly_extracted_discontinued_pred)
        if dataset_name != "Internal Data":
            neither_f1, neither_precision, neither_recall = compute_conditional_metrics(correctly_extracted_neither, correctly_extracted_neither_pred)
            macro_f1 = (active_f1 + discontinued_f1 + neither_f1) / 3
            macro_precision = (active_precision + discontinued_precision + neither_precision) / 3
            macro_recall = (active_recall + discontinued_recall + neither_recall) / 3
        else:
            # Exclude neither_medications for Internal Data
            macro_f1 = (active_f1 + discontinued_f1) / 2
            macro_precision = (active_precision + discontinued_precision) / 2
            macro_recall = (active_recall + discontinued_recall) / 2

        ## Calculate conditional accuracy with original logic
        correct_preds = len(active_acc_count) + len(discontinued_acc_count)
        if dataset_name != "Internal Data":
            correct_preds += len(neither_acc_count)

        acc = correct_preds / len(intersection) if len(intersection) > 0 else 0

        # Update DataFrame with the calculated metrics
        df.loc[index, 'conditional_accuracy'] = acc
        df.loc[index, 'conditional_macro_f1'] = macro_f1
        df.loc[index, 'conditional_macro_precision'] = macro_precision
        df.loc[index, 'conditional_macro_recall'] = macro_recall

    return df

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
    df_w_metrics = calculate_row_metrics(df_w_classifications, dataset_name)

    # Return the final DataFrame with metrics
    return df_w_metrics

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
    df_w_row_metrics = run_pipeline(model_path=model_path, 
                                    input_df=input_df, 
                                    prompt_template=prompt_template, 
                                    dataset_name=name_dataset,
                                    use_sampling=use_sampling,
                                    batch_size=batch_size, 
                                    max_token_output=max_token_output)
    df_w_row_metrics.to_csv(data_folder+f'base_pred_data/{name_dataset}_{model_name}.csv', index=False)

    result_df = pd.read_csv(data_folder+'results.csv')
    metrics_mean = df_w_row_metrics[['extraction_precision', 'extraction_recall', 'conditional_accuracy', 'conditional_macro_f1', 'conditional_macro_precision', 'conditional_macro_recall']].mean(axis=0)

    # Define your result row
    new_row = {
        'Dataset': name_dataset,
        'Model': model_name,
        'Prompt': prompt_template,
        'extraction_precision': metrics_mean.get('extraction_precision', np.nan),
        'extraction_recall': metrics_mean.get('extraction_recall', np.nan),
        'conditional_accuracy': metrics_mean.get('conditional_accuracy', np.nan),
        'conditional_macro_f1': metrics_mean.get('conditional_macro_f1', np.nan),
        'conditional_macro_precision': metrics_mean.get('conditional_macro_precision', np.nan),
        'conditional_macro_recall': metrics_mean.get('conditional_macro_recall', np.nan)
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


