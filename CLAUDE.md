# LLM Benchmarking Framework Guide

## Commands
- Run benchmark: `python src/benchmarks_run.py input_file.jsonl [options]`
- Run with specific options: `python src/benchmarks_run.py input_file.jsonl --output_dir benchmark_results --parallel_calls 4`
- Run prompt optimizer: `python src/prompt_optimizer.py`
- Visualize results: `python src/visualize_results.py`
- Install dependencies: `pip install -r requirements.txt`
- Linting: Use `flake8 src/` for code linting
- Type checking: Not currently configured (could add mypy)

## Code Style Guidelines
- **Imports**: Standard library first, third-party modules second, local modules third, alphabetically within groups
- **Formatting**: Use 4-space indentation, no tabs
- **Variable Naming**: Use snake_case for variables/functions, CamelCase for classes
- **Function Documentation**: Use docstrings with triple quotes explaining purpose, parameters, and return values
- **Error Handling**: Use try/except blocks with specific exceptions, log errors with appropriate severity
- **Logging**: Use the built-in logging module, configure with setup_logging()
- **Types**: Not strictly enforced, but use type hints where applicable
- **File Organization**: Maintain existing module structure with separate files for distinct functionality
- **Comments**: Include descriptive comments for complex logic, separate sections with header comments