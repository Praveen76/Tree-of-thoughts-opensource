import argparse
from tot.methods.bfs import solve
import torch
from tot.models.model_utils import get_model_output
from tot.tasks.game24 import Game24Task, standard_prompt, cot_prompt, propose_prompt

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Set up the core task (Game of 24)
task = Game24Task()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# Loop over each prompting technique and run the experiment for Claude
print(f"\nTesting with CLAUDE backend...\n")

for technique in prompting_techniques:
    print(f"\nRunning experiment with {technique.upper()} prompting technique on CLAUDE...\n")
    
    # Modify the arguments for the current technique and backend
    if technique == 'io':
        args = argparse.Namespace(backend='claude', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'cot':
        args = argparse.Namespace(backend='claude', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'tot':
        args = argparse.Namespace(backend='claude', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

    # Run the experiment for a few records
    for idx in range(2):  # Run for first 2 records as an example
        print(f"\nProcessing record {idx + 1} with {technique.upper()} prompting on CLAUDE...\n")

        # Retrieve the appropriate prompt template based on the technique
        if technique == 'io':
            prompt_template = standard_prompt
        elif technique == 'cot':
            prompt_template = cot_prompt
        elif technique == 'tot':
            prompt_template = propose_prompt

        # Get the input from the task and format the prompt
        input_numbers = " ".join(map(str, task.get_input(idx)))  # Use get_input to standardize input fetching
        prompt = prompt_template.format(input=input_numbers)
        print("prompt:", prompt)

        # Format the prompt for Claude (Human/Assistant turn)
        if args.backend == "claude":
            formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = prompt
        # Use Claude to get output
        claude_output = get_model_output(formatted_prompt, model_name='claude')
        print(f"Claude output for record {idx + 1}: {claude_output}")

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()} bytes")
