import gc
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from baseline import *
import sys

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


def initialize_llm_model(model_path, use_fp16=True, gpu_memory_utilization=0.9):
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
def generate_llm_responses(input_df, batch_size, llm, prompt_template, max_token_output=80, use_sampling=True):
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
    sub_df = input_df['snippet'].values.tolist()

    response_list = []
    num_step = len(sub_df) // batch_size + (1 if len(sub_df) % batch_size != 0 else 0)
    temperature = 0.1 if use_sampling else 0
    top_p = 0.9 if use_sampling else 1
    top_k = 20 if use_sampling else -1

    sampling_params = SamplingParams(
        max_tokens=max_token_output,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    for i in tqdm(range(num_step)):
        input_texts = sub_df[i * batch_size:(i + 1) * batch_size]
        input_texts = [prompt_template.format(text) for text in input_texts]

        # Generate responses with LLM
        responses = llm.generate(input_texts, sampling_params)

        # Process the output
        for response in responses:
            # Extract the generated text and append to list
            response_text = response.outputs[0].text.strip()
            response_list.append(response_text)

    return response_list


# 3. Update run_pipeline to use LLMEngine and align with the original logic
def run_llm_pipeline(model_path, input_df, prompt_template, dataset_name, batch_size=16, max_token_output=80, use_sampling=True):
    """
    Main function to run the text generation pipeline using LLMEngine and compute metrics.
    
    Parameters:
    ----------
    model_path : str
        The path of the model to be used.
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
    # Initialize the model with LLMEngine
    llm_engine = initialize_llm_model(model_path)

    # Generate responses
    response_list = generate_llm_responses(input_df, batch_size, llm_engine, prompt_template, max_token_output, use_sampling)

    # Process the responses to categorize medications
    df_w_classifications = process_output(input_df, response_list, dataset_name)  # Pass dataset_name to keep logic consistent

    # Calculate row-level metrics
    extraction_precision, extraction_recall, extraction_f1, conditional_accuracy, conditional_macro_f1, conditional_macro_precision, conditional_macro_recall = calculate_metrics_by_dataset(df_w_classifications, dataset_name)

    # Return the final DataFrame with metrics
    return df_w_classifications, extraction_precision, extraction_recall, extraction_f1, conditional_accuracy, conditional_macro_f1, conditional_macro_precision, conditional_macro_recall


# 4. Helper function to clear CUDA memory and delete the generator
def clear_cuda_memory_and_terminate(generator=None):
    """Clear the CUDA cache, delete the generator, and run garbage collection before termination."""
    if generator:
        del generator
    torch.cuda.empty_cache()
    gc.collect()

def benchmark_llm_model(name_dataset, model_path, prompt_template, input_df, data_folder, result_df_path, batch_size=16, max_token_output=80, use_sampling=True):
    """
    Function to run the entire LLMEngine pipeline and compute average row-wise metrics for a specific model and dataset.

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
    num_gpus : int
        Number of GPUs to use for LLMEngine.
    batch_size : int
        Number of examples per batch.
    max_token_output : int
        Maximum number of tokens to generate.
    """
    model_name = model_path.split('/')[-1]
    
    # Run the LLMEngine pipeline with dynamic GPU count
    df_w_classifications, extraction_precision, \
    extraction_recall, extraction_f1, conditional_accuracy, \
    conditional_macro_f1, conditional_macro_precision, \
        conditional_macro_recall = run_llm_pipeline(model_path=model_path, 
                                              input_df=input_df, 
                                              prompt_template=prompt_template, 
                                              dataset_name=name_dataset,
                                              batch_size=batch_size, 
                                              max_token_output=max_token_output,
                                              use_sampling=use_sampling)

    # Save the row metrics DataFrame to a CSV
    df_w_classifications.to_csv(data_folder + f'base_pred_data/{name_dataset}_{model_name}.csv', index=False)

    # Read the results CSV
    result_df = pd.read_csv(result_df_path)

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

    # Append the new row to the results DataFrame and save
    result_df = result_df._append(new_row, ignore_index=True).round(3)
    result_df.to_csv(result_df_path, index=False)
    print(f"Results for {model_name} on {name_dataset} saved successfully.")



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
    parser.add_argument('--max_token_output', type=int, default=80, help="Maximum number of tokens to generate.")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse the arguments from the command line
    args = parse_arguments()

    # Load the test dataframe
    test_df = pd.read_csv(os.path.join(args.data_folder, args.test_df_name))


    
    try:
        with open('prompts.json', 'r') as file:
            prompt_template = json.load(file)
        prompt_template = prompt_template[args.prompt_template_key]
        print(f'prompt_template: {prompt_template}')
    except KeyError:
        print(f"Key '{args.prompt_template_key}' not found in the prompt template JSON file.")
        sys.exit(1)

    

    # Run the benchmark using your defined function
    benchmark_llm_model(
        name_dataset=args.dataset_name,
        model_path=args.model_path,
        prompt_template=prompt_template,
        input_df=test_df,
        data_folder=args.data_folder,
        result_df_path=args.result_df_path,
        batch_size=args.batch_size,
        max_token_output=args.max_token_output,
        use_sampling=False
    )

    torch.cuda.empty_cache()
    print(f"Completed inference for model {args.model_path} on dataset {args.dataset_name}")

    sys.exit(0)