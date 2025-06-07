"""Constants for the Streamlit dashboard."""

# App title and information
APP_TITLE = "LLM Benchmarking Dashboard"
SIDEBAR_INFO = """
### LLM Benchmarking Dashboard

This dashboard provides an intuitive interface for:
- Setting up evaluations from CSV files
- Configuring model parameters
- Selecting judge models
- Monitoring evaluation progress
- Viewing results and reports

For more details, see the [README.md](https://github.com/aws-samples/amazon-bedrock-samples/poc-to-prod/360-eval.git)
"""

# Default directories
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_PROMPT_EVAL_DIR = "prompt-evaluations"

# Evaluation parameters
DEFAULT_PARALLEL_CALLS = 4
DEFAULT_INVOCATIONS_PER_SCENARIO = 5
DEFAULT_SLEEP_BETWEEN_INVOCATIONS = 60
DEFAULT_EXPERIMENT_COUNTS = 1
DEFAULT_TEMPERATURE_VARIATIONS = 2

# Default model regions
AWS_REGIONS = [
    "us-east-1", 
    "us-east-2", 
    "us-west-1", 
    "us-west-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "sa-east-1"
]

# Default model list - can be extended
DEFAULT_BEDROCK_MODELS = [
    "amazon.nova-pro-v1:0",
    "amazon.nova-premier-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mistral-large-2402-v1:0"
]

DEFAULT_OPENAI_MODELS = [
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo"
]

# Default token costs (per 1000 tokens)
DEFAULT_COST_MAP = {
    "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "amazon.nova-premier-v1:0": {"input": 0.0025, "output": 0.0125},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
    "meta.llama3-8b-instruct-v1:0": {"input": 0.0001, "output": 0.0002},
    "meta.llama3-70b-instruct-v1:0": {"input": 0.00087, "output": 0.00261},
    "mistral.mistral-7b-instruct-v0:2": {"input": 0.0002, "output": 0.0006},
    "mistral.mistral-large-2402-v1:0": {"input": 0.002, "output": 0.006},
    "openai/gpt-4": {"input": 0.03, "output": 0.06},
    "openai/gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "openai/gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
}