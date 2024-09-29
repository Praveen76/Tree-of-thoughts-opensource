import argparse
from tot.methods.bfs import solve  # Keep this as it is
from tot.models.model_utils import get_model_output  # Import get_model_output
from tot.tasks.game24 import Game24Task

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Set up the core task (Game of 24)
task = Game24Task()

# Loop over each prompting technique and run the experiment for Claude
for technique in prompting_techniques:
    print(f"\nRunning experiment with {technique.upper()} prompting technique on Claude...\n")
    
    # Modify the arguments for the current technique and backend
    if technique == 'io':
        # Simple Input-Output prompting (direct)
        args = argparse.Namespace(backend='claude3-opus', temperature=0.3, task='game24', naive_run=False, 
                                  prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'cot':
        # Chain-of-Thought (step-by-step reasoning)
        args = argparse.Namespace(backend='claude3-opus', temperature=0.3, task='game24', naive_run=False, 
                                  prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'tot':
        # Tree-of-Thoughts (exploring multiple branches of reasoning)
        args = argparse.Namespace(backend='claude3-opus', temperature=0.3, task='game24', naive_run=False, 
                                  prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

    # Run the experiment for a few records
    for idx in range(1):  # Run for first record as an example
        print(f"Processing record {idx + 1} with {technique.upper()} prompting technique...\n")
        ys, info = solve(args, task, idx)
        print(f"Solution: {ys}, Info: {info}")
