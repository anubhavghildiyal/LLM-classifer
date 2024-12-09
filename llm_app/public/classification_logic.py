import openai
from openai import OpenAI

import pdb  # Import the debugger
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def classify_custom(dataset, labels, few_shot_examples, batch_size=10):
    """
    Classify the dataset in batches, with each batch containing a maximum of `batch_size` rows.
    """
    results = []
    num_rows = len(dataset)
    print('in classify_custom')
    client = OpenAI(api_key=openai_api_key)
    # Process the dataset in batches
    for i in range(0, num_rows, batch_size):
        batch = dataset.iloc[i:i + batch_size]
        print('batch created....', i)
        # Construct the prompt for the batch
        batch_prompts = []
        for _, row in batch.iterrows():
            input_text = row["text"]
            batch_prompts.append(input_text)

        # Generate a single prompt for the batch
        formatted_labels = "\n".join(
            [f" {label['id']}: {label['description']}" for i, label in enumerate(labels)]
        )
        prompt = (
            f"Classify the following inputs into one of these labels:\n{formatted_labels}.\n\n"
            "Examples:\n"
        )
        print('prompt created')
        # Add few-shot examples to the prompt
        for example_num, example in enumerate(few_shot_examples):
            prompt += f"Input: {example['input']}\nLabel: {example['label']}\n\n"

        print('few shot examples added')
        # Add batch inputs to the prompt
        prompt += "Inputs:\n"
        for text in batch_prompts:
            prompt += f"Input: {text}\n"

        print('batch inputs added')
        
        # Call the LLM API to classify the batch
        try:
            # Debugger to inspect inputs
            # pdb.set_trace()
            response = client.chat.completions.create(
                # model="gpt-4o-mini",  # Replace with the appropriate model
                model="gpt-4o",  # Replace with the appropriate model
                messages=[
                    {"role": "system", "content": "You are a classification assistant. Classify the following inputs based on the provided labels and descriptions."},
                    {"role": "user", "content": prompt}  # `prompt` contains the input text and few-shot examples
                ],
                max_tokens=5000,  # Adjust based on the expected response length
                temperature=0.1  # Lower temperature for deterministic outputs
            )
            print('response received')
            # Parse LLM's response
            response_text = response.choices[0].message.content.strip()
            print('response parsed')
            # Split the response by double newlines to separate each "Input-Label" pair
            entries = response_text.strip().split("\n\n")

            # Iterate through each entry to extract the input and label
            for entry in entries:
                # Split by newline to separate the input and label lines
                lines = entry.split("\n")
    
                # Extract the input text and label
                input_text = lines[0].replace("Input: ", "").strip()
                label = lines[1].replace("Label: ", "").strip()
                
                # Append the result
                results.append({"input": input_text, "prediction": label})
            print('results appended')
            # pdb.set_trace()

        except Exception as e:
            print(f"Error processing batch: {e}")
            for text in batch_prompts:
                results.append({"input": text, "prediction": "Error"})

        if i == 800:
            break
        # Simulating classification for now
        # batch_results = [{"input": text, "classification": "yes"} for text in batch_prompts]

        # Append results to the main results list
        # results.extend(batch_results)

    return results

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import openai
import time
import threading

# Create a semaphore to control the number of concurrent API calls
RATE_LIMIT = 2  # Max requests per second
semaphore = threading.Semaphore(RATE_LIMIT)

def rate_limited_call(fn, *args, **kwargs):
    """
    Wrapper to enforce rate limiting for LLM API calls.
    """
    with semaphore:
        time.sleep(1 / RATE_LIMIT)  # Wait to respect rate limit
        return fn(*args, **kwargs)

def classify_batch(batch, labels, few_shot_examples, client):
    """
    Function to classify a single batch with rate limiting.
    """
    logging.info(f"Processing batch of size {len(batch)}")
    results = []

    try:
        # Construct the prompt for the batch
        batch_prompts = []
        for _, row in batch.iterrows():
            input_text = row["text"]
            batch_prompts.append(input_text)

        # Generate a single prompt for the batch
        formatted_labels = "\n".join(
            [f" {label['id']}: {label['description']}" for label in labels]
        )
        prompt = (
            f"Classify the following inputs into one of these labels:\n{formatted_labels}.\n\n"
            "Examples:\n"
        )

        # Add few-shot examples to the prompt
        for example_num, example in enumerate(few_shot_examples):
            prompt += f"Input: {example['input']}\nLabel: {example['label']}\n\n"

        # Add batch inputs to the prompt
        prompt += "Inputs:\n"
        for text in batch_prompts:
            prompt += f"Input: {text}\n"

        logging.debug(f"Prompt created:\n{prompt[:500]}...")  # Log first 500 chars of the prompt

        # Call the LLM API (rate-limited)
        response = rate_limited_call(
            client.chat.completions.create,
            model="gpt-4",  # Replace with the appropriate model
            messages=[
                {"role": "system", "content": "You are a classification assistant. Classify the following inputs based on the provided labels and descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5000,  # Adjust based on the expected response length
            temperature=0.1  # Lower temperature for deterministic outputs
        )

        # Parse LLM's response
        response_text = response.choices[0].message.content.strip()
        logging.info("Response received from LLM.")
        
        # Split the response by double newlines to separate each "Input-Label" pair
        entries = response_text.strip().split("\n\n")

        # Iterate through each entry to extract the input and label
        for entry in entries:
            lines = entry.split("\n")
            input_text = lines[0].replace("Input: ", "").strip()
            label = lines[1].replace("Label: ", "").strip()
            results.append({"input": input_text, "prediction": label})

        logging.info(f"Batch processed successfully with {len(results)} results.")

    except Exception as e:
        logging.error(f"Error processing batch: {e}", exc_info=True)
        for text in batch_prompts:
            results.append({"input": text, "prediction": "Error"})

    return results

def classify_custom_parallel(dataset, labels, few_shot_examples, batch_size=10, num_workers=2):
    """
    Classify the dataset in parallel using ThreadPoolExecutor with rate limiting.
    """
    logging.info("Starting parallel classification.")
    results = []
    num_rows = len(dataset)
    client = openai  # Initialize OpenAI client

    # Create batches
    batches = [dataset.iloc[i:i + batch_size] for i in range(0, num_rows, batch_size)]
    logging.info(f"Dataset divided into {len(batches)} batches with batch size {batch_size}.")

    # Use ThreadPoolExecutor for parallel batch processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(classify_batch, batch, labels, few_shot_examples, client): batch
            for batch in batches[:5]  # Limit for demonstration
        }

        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                logging.info(f"Batch completed. Total results collected: {len(results)}")
            except Exception as e:
                logging.error(f"Error in batch processing: {e}", exc_info=True)

    logging.info("All batches processed successfully.")
    return results
