import gc
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import sys
import os
import json
import pandas as pd
import argparse
from ast import literal_eval
import re

def get_visible_gpus():
    """
    Get the number of visible GPUs based on the CUDA_VISIBLE_DEVICES environment variable.
    
    Returns:
    -------
    num_gpus : int
        The number of GPUs visible in the environment.
    """
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        num_gpus = len(cuda_visible_devices.split(","))
    else:
        num_gpus = torch.cuda.device_count()  # Fallback to all GPUs if not explicitly set
    return num_gpus


def initialize_llm_model(model_path, use_fp16=True, gpu_memory_utilization=0.8):
    """
    Initializes the model for text generation using LLM on dynamically detected GPUs with dynamically set KV cache.

    Parameters:
    ----------
    model_path : str
        Path of the model to be loaded.
    use_fp16 : bool
        Whether to use FP16 precision for reduced memory usage.
    gpu_memory_utilization : float
        Fraction of GPU memory to use for the KV cache.
    
    Returns:
    -------
    llm : LLM
        An LLM instance configured for text generation.
    """
    num_gpus = get_visible_gpus()  # Dynamically detect number of GPUs

    # Set the precision type
    dtype = torch.float16 if use_fp16 else torch.float32

    # Initialize LLM with the calculated max_model_len
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=num_gpus, 
        dtype=dtype,
        max_model_len=4096 if 'meditron-7b' not in model_path else None,  
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    return llm

# 2. Function to generate batch responses using the model
def generate_llm_responses(input_df, batch_size, llm, prompt_template, max_token_output=80, use_sampling=True, with_groundtruth=False):
    """
    Generate text responses in batches using LLM.
    
    Parameters:
    ----------
    input_df : pd.DataFrame
        DataFrame containing input texts for inference.
    batch_size : int
        Size of each batch for inference.
    llm : LLM
        LLM instance initialized for text generation.
    prompt_template : str
        Template for constructing prompts.
    max_token_output : int
        Maximum number of tokens to generate.
    use_sampling : bool
        Whether to use sampling or greedy decoding.
    
    Returns:
    -------
    response_list : list of str
        List of generated responses.
    """
    if with_groundtruth:
        p_list = prompt_template.split("\nOutput:") 

        # insert to the second last element
        insert = "Hint: Here is a complete list of medications included in this note: {}. Assign a status for each of them.\n"
        # rejoin the prompt template to have the option to insert the hint
        prompt_template = '\nOutput:'.join(p_list[:-1]) + insert + "\nOutput:" + p_list[-1]

    sub_df = input_df['snippet'].values.tolist()

    response_list = []
    num_step = len(sub_df) // batch_size + (1 if len(sub_df) % batch_size != 0 else 0)
    temperature = 0.1 if use_sampling else 0
    top_p = 0.9 if use_sampling else 1
    # top_k = 20 if use_sampling else -1

    sampling_params = SamplingParams(
        max_tokens=max_token_output,
        temperature=temperature,
        top_p=top_p,
        # top_k=top_k
    )

    for i in tqdm(range(num_step)):
        input_texts = sub_df[i * batch_size:(i + 1) * batch_size]
        if with_groundtruth:
            ground_truth_list = input_df['true_set'][i * batch_size:(i + 1) * batch_size]
            input_texts = [prompt_template.format(text, ground_truth) for text, ground_truth in zip(input_texts, ground_truth_list)]
        else:
            input_texts = [prompt_template.format(text) for text in input_texts]

        # Generate responses with LLM
        responses = llm.generate(input_texts, sampling_params)

        # Process the output
        for response in responses:
            # Extract the generated text and append to list
            response_text = response.outputs[0].text.strip()
            response_list.append(response_text)

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

    updated_response_list = []
    for response in response_list:
        response = response.split("END")[0].strip()
        updated_response_list.append(response)
        medications = re.findall(r'- (.*?) \((active|discontinued|neither)\)', response, re.IGNORECASE)

        # Initialize lists for each medication status
        active_medications = []
        discontinued_medications = []
        neither_medications = []

        # Group medications based on their status
        for med_name, status in medications:
            if re.search(r'[a-zA-Z0-9]', med_name) and med_name.strip().lower() not in ['none', 'n/a', 'na', 'unknown']:
                med_name = med_name.strip().lower()
                if status.lower() == 'active':
                    active_medications.append(med_name)
                elif status.lower() == 'discontinued':
                    discontinued_medications.append(med_name)
                elif status.lower() == 'neither' and dataset_name != "Internal Data":
                    neither_medications.append(med_name)

        # Append each category list to their respective main lists
        active_medications_list.append(active_medications)
        discontinued_medications_list.append(discontinued_medications)
        if dataset_name != "Internal Data":
            neither_medications_list.append(neither_medications)

    # Create a new DataFrame from the lists
    output_df = pd.DataFrame({
        'model_response': updated_response_list,
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

    def update_pred_that_contains_true_med_name(pred_list, true_list):
        pred_list_copy = pred_list.copy()  # Make a copy to iterate safely
        updated_med_list = []
        for pred_med in pred_list_copy:
            matched = False  # Track if we find a match for the current pred_med
            for true_med in true_list:
                if re.search(true_med, pred_med, re.IGNORECASE):
                    updated_med_list.append(true_med)
                    pred_list.remove(pred_med)  # Modify original pred_list here
                    matched = True
                    break  # Exit the inner loop since we found a match
            if not matched:
                updated_med_list.append(pred_med)  # Append pred_med as it is if no match is found

        return updated_med_list

    col_list = ['active_medications', 'discontinued_medications', 'neither_medications'] if dataset_name != "Internal Data" else ['active_medications', 'discontinued_medications']
    for col in col_list:
        updated_col = []
        for row in result_df.iterrows():
            true_set = row[1]['true_set']
            updated_pred = update_pred_that_contains_true_med_name(row[1][f'{col}_pred'], true_set)
            updated_pred = list(set(updated_pred))
            updated_col.append(updated_pred)
        result_df[f'{col}_pred'] = updated_col
 
    return result_df

# 4. Function to calculate metrics (Precision, Recall, F1, Accuracy)
# Function to calculate extraction metrics (Precision, Recall, F1)
def calculate_extraction_metrics(df, dataset_name):
    # Combine true and predicted sets
    col_list = ['active_medications', 'discontinued_medications'] if dataset_name == 'Internal Data' else ['active_medications', 'discontinued_medications', 'neither_medications']
    pred_col_list = [f'{col}_pred' for col in col_list]
    df['pred_set'] = df[pred_col_list].apply(lambda x: set([med for meds in x for med in meds]), axis=1)

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

    return df, extraction_precision, extraction_recall, extraction_f1

# Function to calculate classification metrics (Precision, Recall, F1, Accuracy)
def calculate_classification_metrics(df, dataset_name):
    
    def calculate_joint_metrics(df, med_class):
        # Define columns dynamically based on the class
        true_col = f'{med_class}_medications'
        pred_col = f'{med_class}_medications_pred'

        joint_pred_count_col = f'joint_{med_class}_pred_count'
        joint_true_count_col = f'joint_{med_class}_true_count'
        joint_intersection_count_col = f'joint_{med_class}_intersection_count'
        

        # Calculate counts for task 2 precision and recall
        df[joint_pred_count_col] = df[pred_col].apply(lambda x: len(x))
        df[joint_true_count_col] = df[true_col].apply(lambda x: len(x))
        df[joint_intersection_count_col] = df.apply(lambda row: len(set(row[true_col]).intersection(set(row[pred_col]))), axis=1)
        
        # Calculate precision, recall, and F1 for the given class
        precision = df[joint_intersection_count_col].sum() / df[joint_pred_count_col].sum() if df[joint_pred_count_col].sum() > 0 else 0
        recall = df[joint_intersection_count_col].sum() / df[joint_true_count_col].sum() if df[joint_true_count_col].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
        return df, precision, recall, f1

    if 'intersection' not in df.columns:
        df['intersection'] = df['true_set']
        df['true_count'] = df['true_set'].apply(lambda x: len(x))
        df['pred_count'] = df['true_count']

    # Apply task2 metrics calculation for each medication class
    df, active_precision, active_recall, active_f1 = calculate_joint_metrics(df, 'active')
    df, discontinued_precision, discontinued_recall, discontinued_f1 = calculate_joint_metrics(df, 'discontinued')

    if dataset_name != 'Internal Data':
        df, neither_precision, neither_recall, neither_f1 = calculate_joint_metrics(df, 'neither')
    else:
        neither_precision, neither_recall, neither_f1 = 0, 0, 0

    # Add a column to sum the correct predictions for the 3 classes
    df['correct_pred_count'] = df['joint_active_intersection_count'] + df['joint_discontinued_intersection_count'] + (df['joint_neither_intersection_count'] if dataset_name != 'Internal Data' else 0)

    # Calculate the macro metrics
    joint_accuracy = df['correct_pred_count'].sum() / df['pred_count'].sum() if df['true_count'].sum() > 0 else 0   
    joint_macro_f1 = (active_f1 + discontinued_f1 + neither_f1) / (2 if dataset_name == 'Internal Data' else 3)
    joint_macro_precision = (active_precision + discontinued_precision + neither_precision) / (2 if dataset_name == 'Internal Data' else 3)
    joint_macro_recall = (active_recall + discontinued_recall + neither_recall) / (2 if dataset_name == 'Internal Data' else 3)

    return joint_accuracy, joint_macro_f1, joint_macro_precision, joint_macro_recall

# Main function to calculate metrics by dataset
def calculate_metrics_by_dataset(df, dataset_name):
    # Calculate extraction metrics
    df, extraction_precision, extraction_recall, extraction_f1 = calculate_extraction_metrics(df, dataset_name)

    # Calculate classification metrics
    joint_accuracy, joint_macro_f1, joint_macro_precision, joint_macro_recall = calculate_classification_metrics(df, dataset_name)

    return extraction_precision, extraction_recall, extraction_f1, joint_accuracy, joint_macro_f1, joint_macro_precision, joint_macro_recall

# 3. Update run_pipeline to use LLMEngine and align with the original logic
def run_llm_pipeline(llm_model, input_df, prompt_template, dataset_name, batch_size=16, max_token_output=80, use_sampling=True):
    """
    Main function to run the text generation pipeline using LLMEngine and compute metrics.
    
    Parameters:
    ----------
    llm_model : str
        The initilized llm model.
    input_df : pd.DataFrame
        The data to be inferred.
    prompt_template : str
        Template for constructing the prompts.
    dataset_name : str
        Name of the dataset. Used for special cases like Internal Data exclusion.
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
    # Generate responses
    response_list = generate_llm_responses(input_df, batch_size, llm_model, prompt_template, max_token_output, use_sampling, with_groundtruth=False)
    # generate responses with ground truth list
    response_list_with_groundtruth = generate_llm_responses(input_df, batch_size, llm_model, prompt_template, max_token_output, use_sampling, with_groundtruth=True)

    # Process the responses to categorize medications
    df_w_classifications = process_output(input_df, response_list, dataset_name)  
    # process the responses with ground truth to categorize medications
    df_w_classifications_with_groundtruth = process_output(input_df, response_list_with_groundtruth, dataset_name)

    # Calculate row-level metrics
    extraction_precision, extraction_recall, extraction_f1, joint_accuracy, joint_macro_f1, joint_macro_precision, joint_macro_recall = calculate_metrics_by_dataset(df_w_classifications, dataset_name)
    # calculate the classification metrics with responses with ground truth
    accuracy, macro_f1, macro_precision, macro_recall = calculate_classification_metrics(df_w_classifications_with_groundtruth, dataset_name)

    # append the active_medications_pred, discontinued_medications_pred, neither_medications_pred to the df_w_classifications with extension of _with_groundtruth
    df_w_classifications_with_groundtruth = df_w_classifications_with_groundtruth[['model_response', 'active_medications_pred', 'discontinued_medications_pred', 'neither_medications_pred']] if dataset_name != "Internal Data" else df_w_classifications_with_groundtruth[['model_response', 'active_medications_pred', 'discontinued_medications']]
    df_w_classifications_with_groundtruth.columns = [f'{col}_with_groundtruth' for col in df_w_classifications_with_groundtruth.columns]
    df_w_classifications = pd.concat([df_w_classifications, df_w_classifications_with_groundtruth], axis=1)

    # Return the final DataFrame with metrics
    return df_w_classifications, extraction_precision, extraction_recall, extraction_f1, joint_accuracy, joint_macro_f1, joint_macro_precision, joint_macro_recall, accuracy, macro_f1, macro_precision, macro_recall


# 4. Helper function to clear CUDA memory and delete the generator
def clear_cuda_memory_and_terminate(generator=None):
    """Clear the CUDA cache, delete the generator, and run garbage collection before termination."""
    if generator:
        del generator
    torch.cuda.empty_cache()
    gc.collect()


def benchmark_llm_model(dataset_name, model_path, prompt_templates, input_df, data_folder, result_df_path, batch_size=16, max_token_output=80, use_sampling=False, one_shot=False, cot=False, five_shot=False):
    """
    Function to run the entire LLMEngine pipeline and compute average row-wise metrics for a specific model and dataset.

    Parameters:
    ----------
    dataset_name : str
        The name of the dataset being processed (e.g., "Internal Data" or "MIT").
    model_path : str
        The path of the model to be used.
    prompt_templates : list of str
        List of templates for constructing the prompts.
    input_df : pd.DataFrame
        Data to run inference on.
    data_folder : str
        The folder where input data and results are stored.
    result_df_path : str
        Path to the CSV file where results will be stored.
    batch_size : int
        Number of examples per batch.
    max_token_output : int
        Maximum number of tokens to generate.
    one_shot : bool
        Whether the current run is a one-shot test.
    """
    model_name = model_path.split('/')[-1]
    # check if input_df.active_medications[0] is a list, if not, apply eval and lower to active_medications, discontinued_medications, neither_medications
    if not isinstance(input_df.active_medications[0], list):
        input_df.loc[:, 'active_medications'] = input_df['active_medications'].apply(lambda x: literal_eval(x)).apply(lambda x: [med.lower() for med in x])
        input_df.loc[:, 'discontinued_medications'] = input_df['discontinued_medications'].apply(lambda x: literal_eval(x)).apply(lambda x: [med.lower() for med in x])
        if dataset_name != "Internal Data":
            input_df.loc[:, 'neither_medications'] = input_df['neither_medications'].apply(lambda x: literal_eval(x)).apply(lambda x: [med.lower() for med in x])
    
    col_list = ['active_medications', 'discontinued_medications', 'neither_medications'] if dataset_name != "Internal Data" else ['active_medications', 'discontinued_medications']
    input_df['true_set'] = input_df[col_list].apply(lambda x: set([med for meds in x for med in meds]), axis=1)
    
    # Initialize the model with LLMEngine
    llm_engine = initialize_llm_model(model_path)
    
    # Iterate over each prompt template
    for sim in range(1, 6):
        for i, prompt_template in enumerate(prompt_templates):
            print(f"Running with prompt template {i+1}/{len(prompt_templates)}")
            
            # Run the LLMEngine pipeline with dynamic GPU count
            df_w_classifications, extraction_precision, \
            extraction_recall, extraction_f1, joint_accuracy, \
            joint_macro_f1, joint_macro_precision, \
            joint_macro_recall, \
            accuracy, macro_f1, macro_precision, macro_recall = run_llm_pipeline(
                llm_model=llm_engine, 
                input_df=input_df, 
                prompt_template=prompt_template, 
                dataset_name=dataset_name,
                batch_size=batch_size, 
                max_token_output=max_token_output,
                use_sampling=use_sampling
            )

            # Save the row metrics DataFrame to a CSV
            if cot and one_shot:
                output_filename = f'{dataset_name}_{model_name}_cot_1_shot_{i+1}_cot_sim_{sim}.csv'
            elif cot and five_shot:
                output_filename = f'{dataset_name}_{model_name}_5_shot_cot_sim_{sim}.csv'
            elif cot:
                output_filename = f'{dataset_name}_{model_name}_cot_sim_{sim}.csv'
            elif one_shot:
                output_filename = f'{dataset_name}_{model_name}_1_shot_{i+1}_sim_{sim}.csv'
            elif five_shot:
                output_filename = f'{dataset_name}_{model_name}_5_shots_sim_{sim}.csv'
            else:
                output_filename = f'{dataset_name}_{model_name}_sim_{sim}.csv'

            df_w_classifications.to_csv(data_folder + f'base_pred_data/{output_filename}', index=False)

            # Read the results CSV
            result_df = pd.read_csv(result_df_path)

            # Define your result row
            new_row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Prompt': prompt_template,
                'Simulation': sim,

                'extraction_precision': extraction_precision,
                'extraction_recall': extraction_recall,
                'extraction_f1': extraction_f1,

                'accuracy_w_gt': accuracy,
                'macro_f1_w_gt': macro_f1,
                'macro_precision_w_gt': macro_precision,
                'macro_recall_w_gt': macro_recall,

                'joint_accuracy': joint_accuracy,
                'joint_macro_f1': joint_macro_f1,
                'joint_macro_precision': joint_macro_precision,
                'joint_macro_recall': joint_macro_recall,
            }

            # Append the new row to the results DataFrame and save
            result_df = result_df._append(new_row, ignore_index=True).round(3)
            result_df.to_csv(result_df_path, index=False)
            print(f"Simulation {sim} results for {model_name} on {dataset_name} with prompt {i+1} saved successfully.")


# Function to parse command-line arguments
def parse_arguments():
    """Parse command-line arguments passed from the shell script."""
    parser = argparse.ArgumentParser(description="LLM Benchmark Inference Pipeline")
    
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model to be used.")
    parser.add_argument('--test_df_name', type=str, required=True, help="Name of the test dataframe (CSV file).")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset being processed (e.g., 'Internal Data').")
    parser.add_argument('--prompt_template_key', type=str, required=True, help="Key for the prompt template in the JSON file.")
    parser.add_argument('--data_folder', type=str, required=True, help="Path to the data folder containing test data and saving results.")
    parser.add_argument('--result_df_path', type=str, required=True, help="Path to the CSV file where results will be stored.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference.")
    parser.add_argument('--one_shot', type=str, required=True, help="Whether to perform one-shot inference.")
    parser.add_argument('--cot', type=str, required=True, help="Whether to perform chain-of-thoughts inference.")
    parser.add_argument('--five_shot', type=str, required=True, help="Whether to perform five-shot inference.")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse the arguments from the command line
    args = parse_arguments()

    # Load the test dataframe
    test_df = pd.read_csv(os.path.join(args.data_folder, args.test_df_name))

    try:
        with open('prompts.json', 'r') as file:
            prompt_templates_json = json.load(file)

        prompt_template_keys = [args.prompt_template_key]
        if args.one_shot == 'true' and args.cot == 'true':
            prompt_template_keys = [f"{args.prompt_template_key}_1_shot_{i+1}_CoT" for i in range(5)]
        elif args.five_shot == 'true' and args.cot == 'true':
            prompt_template_keys = [f"{args.prompt_template_key}_5_shots_CoT"]
        elif args.cot == 'true':
            prompt_template_keys = [f"{args.prompt_template_key}_CoT"]
        elif args.one_shot == 'true':
            prompt_template_keys = [f"{args.prompt_template_key}_1_shot_{i+1}" for i in range(5)]
        elif args.five_shot == 'true':
            prompt_template_keys = [f"{args.prompt_template_key}_5_shots"]

        prompt_templates = [prompt_templates_json[key] for key in prompt_template_keys]
        print(f'Loaded prompt key {prompt_template_keys} for inference.')
    except KeyError as e:
        print(f"Key error occurred: {e}")
        sys.exit(1)

    # Run the benchmark using your defined function
    benchmark_llm_model(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        prompt_templates=prompt_templates,
        input_df=test_df,
        data_folder=args.data_folder,
        result_df_path=args.result_df_path,
        batch_size=args.batch_size,
        max_token_output=80 if args.cot != 'true' else 300,
        use_sampling=True,
        one_shot=True if args.one_shot == 'true' else False,
        cot=True if args.cot == 'true' else False,
        five_shot=True if args.five_shot == 'true' else False
    )

    torch.cuda.empty_cache()
    print(f"Completed inference for model {args.model_path} on dataset {args.dataset_name}")

    sys.exit(0)