import subprocess

# Define the tasks to test
tasks = [
    "boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_challenge", "arc_easy",
    "obqa", "copa", "race_middle", "race_high", "mmlu", "nq", "tqa", "math",
    "gsm8k", "humaneval", "mbpp", "the_stack", "books", "wikipedia"
]

# Model and evaluation parameters
model = "facebook/layerskip-llama2-7B"
generation_strategy = "autoregressive"
exit_layer = 8
output_dir = "./logs"
limit = 1  # Number of samples to test per task

# Iterate through all tasks and execute the evaluation command
for task in tasks:
    print(f"Running evaluation for task: {task}")
    command = [
        "torchrun", "eval.py",
        "--model", model,
        "--tasks", task,
        "--limit", str(limit),
        "--generation_strategy", generation_strategy,
        "--exit_layer", str(exit_layer),
        "--output_dir", output_dir
    ]
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Task {task} completed successfully.\nOutput:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        # Log errors if the task fails
        print(f"Task {task} failed.\nError:\n{e.stderr}")
