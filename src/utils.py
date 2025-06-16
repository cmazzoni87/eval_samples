import pytz
import datetime
import os
import boto3
import time
import logging
import random
import json
import tiktoken
import re
from google import genai
from litellm import completion
from botocore.exceptions import ClientError
from botocore.config import Config
from google.genai.types import HttpOptions, Part

logger = logging.getLogger(__name__)


# ----------------------------------------
# Request Builders
# ----------------------------------------
def get_body(prompt, max_tokens, temperature, top_p):
    sys = ""
    body = [{"role": "user", "content": [{"text": f"{sys}\n##USER:{prompt}"}]}]
    cfg  = {"maxTokens": max_tokens, "temperature": temperature, "topP": top_p}
    return body, cfg


def setup_logging(log_dir='logs', experiment='none'):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/360-benchmark-{ts}-{experiment}.log"
    
    # Reset root logger and handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure root logger
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # Add console handler for warnings and above
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)
    
    # Configure logger for this module
    module_logger = logging.getLogger(__name__)
    module_logger.info(f"Logging initialized. Log file: {log_file}")
    
    return ts, log_file


def get_bedrock_client(region):
    cfg = Config(retries={"max_attempts": 10})
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time(), tz=pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def calculate_average_scores(dict_list):
    if not dict_list:
        logger.warning("Attempted to calculate average scores with empty list")
        return {}

    # Initialize result dictionary
    result = {}
    
    # Validate input
    valid_dicts = []
    for i, d in enumerate(dict_list):
        if not isinstance(d, dict):
            logger.warning(f"Non-dictionary item at index {i} in score list: {type(d)}")
            continue
        if not d:
            logger.warning(f"Empty dictionary at index {i} in score list")
            continue
        valid_dicts.append(d)
    
    if not valid_dicts:
        logger.warning("No valid dictionaries found for score calculation")
        return {}

    # Get the keys from the first dictionary
    keys = valid_dicts[0].keys()

    # Calculate the average for each key
    for key in keys:
        try:
            values = []
            for d in valid_dicts:
                if key in d:
                    try:
                        # Only include numeric values
                        value = d[key]
                        if isinstance(value, (int, float)):
                            values.append(value)
                        else:
                            logger.warning(f"Non-numeric value '{value}' for key '{key}' in score calculation")
                    except Exception as e:
                        logger.warning(f"Error processing value for key '{key}': {e}")
            
            if values:
                average = sum(values) / len(values)
                result[f'AVG_{key}'] = round(average, 4)
            else:
                logger.warning(f"No valid values found for key '{key}' in score calculation")
                result[f'AVG_{key}'] = None
        except Exception as e:
            logger.error(f"Error calculating average for key '{key}': {e}")
            result[f'AVG_{key}'] = None
            
    return result


def converse_with_bedrock(
        region,
        model_id,
        messages,
        inference_config,
        perf_config="standard",
        max_retries=10,
        initial_backoff=1,
        max_backoff=300,  # 5 minutes in seconds
        jitter=True,
        stream=True,
):

    # Initialize Bedrock client
    bedrock_runtime = get_bedrock_client(region)

    attempts = 0
    backoff_time = initial_backoff
    # request_count = 1
    while attempts <= max_retries:
        try:
            if stream:
                ttft = time.time()
                return bedrock_runtime.converse_stream(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig=inference_config,
                    performanceConfig=perf_config
                ), ttft, attempts + 1
            else:
                return bedrock_runtime.converse(
                    messages=messages,
                    modelId=model_id,
                    inferenceConfig=inference_config), None, attempts + 1

        except ClientError as error:
            error_code = error.response['Error']['Code']
            error_message = error.response['Error']['Message']

            # Handle throttling errors with exponential backoff
            if error_code in ['ThrottlingException', 'TooManyRequestsException', 'ServiceUnavailableException', 'ModelErrorException']:
                # request_count += 1
                if attempts < max_retries:
                    # Apply jitter to backoff time if enabled
                    if jitter:
                        actual_backoff = backoff_time * (0.5 + random.random())
                    else:
                        actual_backoff = backoff_time

                    # Cap the backoff time at max_backoff
                    actual_backoff = min(actual_backoff, max_backoff)
                    if stream:
                        logger.warning(
                            f"Throttling error encountered (attempt {attempts + 1}/{max_retries}) with Model: {model_id}. "
                            f"Backing off for {actual_backoff:.2f} seconds."
                        )
                    else:
                        logger.warning(
                            f"Throttling error encountered (attempt {attempts + 1}/{max_retries}) with Judge Model: {model_id}. "
                            f"Backing off for {actual_backoff:.2f} seconds."
                        )

                    time.sleep(actual_backoff)

                    # Increase backoff for next attempt
                    backoff_time = min(backoff_time * 2, max_backoff)
                    attempts += 1
                else:
                    logger.error(f"Maximum retry attempts reached after throttling. Last error: {error_message}")
                    raise
            else:
                # If not a throttling error, re-raise
                logger.error(f"Bedrock error: {error_code} - {error_message}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    # If we've exhausted all retries
    raise Exception(f"Failed to get a response after {max_retries} retry attempts")


def extract_json_with_llm(all_metrics, text, judge_model_id, judge_region, cfg):
    metrics_entries = [f'            "{metric}": <int>' for metric in all_metrics]
    metrics_string = ",\n".join(metrics_entries)

    prompt = f"""## Instruction
Extract and return the JSON object from the given text that matches the specified JSON schema. The schema is:
```json
{{
    "scores": {{
{metrics_string}
            }}
}}
```
## Text
{text}

Provide your response immediately without any preamble or additional information.
            """
    body = [{"role": "user", "content": [{"text": prompt}]}]
    resp, = converse_with_bedrock(messages=body,
                                  model_id=judge_model_id,
                                  region=judge_region,
                                  inference_config=cfg,
                                  stream=False)
    text = resp['output']['message']['content'][0]['text']
    payload = extract_json_from_text(text)
    if not payload:
        return None
    return payload


def extract_json_from_text(text):
    pattern = re.compile(
        r'\{\s*"scores"\s*:\s*\{\s*(?:[^{}]*?)\s*\}\s*\}',
        re.VERBOSE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        json_text = match.group(0)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Found JSON block but failed to parse: {e}")
            return None
    else:
        logger.error("No matching JSON block found in response")
        return None


def extract_json_response(all_metrics, text, judge_model_id, judge_region, cfg):
    payload = extract_json_from_text(text)
    if not payload:
        logger.warning(f"Initial JSON extraction failed for {judge_model_id}, attempting extraction with LLM")
        payload = extract_json_with_llm(all_metrics, text, judge_model_id, judge_region, cfg)
        if not payload:
            logger.error(f"LLM-based JSON extraction also failed for {judge_model_id}")
    return payload



def llm_judge_template(all_metrics,
                       task_types,
                       task_criteria,
                       prompt,
                       model_response,
                       golden_answer
                       ):

    metrics_list = "\n".join(f"- {m}" for m in all_metrics)
    metrics_entries = [f'            "{metric}": <int>' for metric in all_metrics]
    metrics_string = ",\n".join(metrics_entries)
    return f"""
    ## You are an expert evaluator.  
    # Task: {task_types}

    # Task description: {task_criteria}

    # Original Prompt:
    {prompt}

    # Model Response:
    {model_response}

    # Golden (Reference) Response:
    {golden_answer}

    # Please evaluate the model response on the following metrics:
    {metrics_list}

    # For each metric, assign an integer score from 1 (worst) to 5 (best).

    ## IMPORTANT: **Output JSON only** in this format:
    ```json
    {{
      "scores": {{
{metrics_string}
      }}
    }}
    ```
    """.strip()


# Count tokens using tiktoken
def count_oai_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


def count_gcp_tokens(provider_client, text: str) -> int:
    return provider_client.models.count_tokens(model="gemini-1.5-flash", contents=text).total_tokens


# Run streaming inference and collect metrics
def run_3p_inference(model_name: str,
                     prompt_text: str,
                     _input_cost:float,
                     _output_cost:float,
                     provider_params: dict):

    # Concatenate user prompt for token counting
    messages = [{ "content": prompt_text, "role": "user"}]
    start_time = time.time()
    response_chunks = []
    first = True
    time_to_first_token = 0
    if 'gemini' in model_name:
        os.environ['GEMINI_API_KEY'] = provider_params['api_key']
        provider_params = {}

    for chunk in completion(
        model=model_name,
        messages=messages,
        stream=True,
        **provider_params
    ):
        if first:
            time_to_first_token = time.time() - start_time
            first = False
        delta = chunk.choices[0].delta.get("content", "")
        if delta:
            response_chunks.append(delta)

    total_runtime = time.time() - start_time
    full_response = "".join(response_chunks)
    if 'openai' in model_name:
        output_tokens = count_oai_tokens(full_response)
        input_tokens = count_oai_tokens(prompt_text)
    elif 'gemini' in model_name:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        output_tokens = count_gcp_tokens(client, full_response)
        input_tokens = count_gcp_tokens(client, prompt_text)
    else:
        # AZURE FOR NOW
        output_tokens = count_oai_tokens(full_response)
        input_tokens = count_oai_tokens(prompt_text)

    tokens_per_sec = output_tokens / total_runtime
    input_cost = input_tokens * (_input_cost / 1000)
    output_cost = output_tokens * (_output_cost / 1000)


    return {
        "time_to_first_byte": time_to_first_token,
        "time_to_last_byte": total_runtime,
        "throughput_tps": tokens_per_sec,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_cost": input_cost + output_cost,
        "model_response": full_response
    }
