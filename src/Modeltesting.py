import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Set up the core task (Game of 24 in this case)
task = Game24Task()

# Loop over each prompting technique and run the experiment
for technique in prompting_techniques:
    print(f"\nRunning experiment with {technique.upper()} prompting technique...\n")
    
    # Modify the arguments for the current technique
    if technique == 'io':
        # Simple Input-Output prompting (direct)
        args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'cot':
        # Chain-of-Thought (step-by-step reasoning)
        args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'tot':
        # Tree-of-Thoughts (exploring multiple branches of reasoning)
        args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

    # Run the experiment for a few records (adjust idx for more samples)
    for idx in range(2):  # Run for first 3 records as an example
        print(f"\nProcessing record {idx + 1} with {technique.upper()}...\n")
        ys, infos = solve(args, task, idx)
        print(f"Result for record {idx + 1}: {ys[0]}")  # Print the first result for each record

