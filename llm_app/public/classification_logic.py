import openai
from openai import OpenAI

import pdb  # Import the debugger
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def classify_custom(dataset, labels, few_shot_examples, batch_size=10, max_retries=3):
    """
    Classify the dataset in batches, with each batch containing a maximum of `batch_size` rows.
    If predictions are missing or invalid, the batch is retried up to `max_retries` times.
    """
    results = []
    num_rows = len(dataset)
    print('in classify_custom')
    client = OpenAI(api_key=openai_api_key)
    # Process the dataset in batches
    for i in range(0, num_rows, batch_size):
        batch = dataset.iloc[i:i + batch_size]
        print('batch created....', i)
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
        print('prompt created')
        for example in few_shot_examples:
            prompt += f"Input: {example['input']}\nLabel: {example['label']}\n\n"

        print('few shot examples added')
        prompt += "Inputs:\n"
        for text in batch_prompts:
            prompt += f"Input: {text}\n"

        print('batch inputs added')

        # Retry mechanism for invalid predictions
        retries = 0
        while retries < max_retries:
            try:
                # Call the LLM API to classify the batch
                response = client.chat.completions.create(
                    model="gpt-4o",  # Replace with the appropriate model
                    messages=[
                        {"role": "system", "content": "You are a classification assistant. Classify the following inputs based on the provided labels and descriptions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=5000,
                    temperature=0.1
                )
                print('response received')

                # Parse LLM's response
                response_text = response.choices[0].message.content.strip()
                print('response parsed')

                # Split the response by double newlines to separate each "Input-Label" pair
                entries = response_text.strip().split("\n\n")

                # Validate predictions
                if len(entries) != len(batch_prompts):
                    raise ValueError("Mismatch in number of predictions. Retrying batch.")

                # Iterate through each entry to extract the input and label
                for entry in entries:
                    lines = entry.split("\n")
                    input_text = lines[0].replace("Input: ", "").strip()
                    label = lines[1].replace("Label: ", "").strip()

                    # Check if label is valid
                    if label not in [str(label['id']) for label in labels]:
                        raise ValueError(f"Invalid label: {label}. Retrying batch.")

                    # Append the result
                    results.append({"input": input_text, "prediction": label})

                print('results appended')
                break  # Exit retry loop if successful

            except Exception as e:
                retries += 1
                print(f"Error processing batch (Attempt {retries}/{max_retries}): {e}")
                if retries == max_retries:
                    print("Max retries reached. Marking predictions as 'Error'.")
                    for text in batch_prompts:
                        results.append({"input": text, "prediction": "Error"})

        if i == 1000:
            break

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
