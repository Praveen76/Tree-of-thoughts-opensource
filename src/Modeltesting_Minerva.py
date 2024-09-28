import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import torch
from tot.models.model_utils import get_model_output  # Import get_model_output

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Set up the core task (Game of 24)
task = Game24Task()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# Loop over each prompting technique and run the experiment for Minerva
print(f"\nTesting with MINERVA backend...\n")

for technique in prompting_techniques:
    print(f"\nRunning experiment with {technique.upper()} prompting technique on MINERVA...\n")
    
    # Modify the arguments for the current technique and backend
    if technique == 'io':
        args = argparse.Namespace(backend='minerva', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'cot':
        args = argparse.Namespace(backend='minerva', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    elif technique == 'tot':
        args = argparse.Namespace(backend='minerva', temperature=0.7, task='game24', naive_run=False, 
                                  prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                  method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

    # Run the experiment for a few records
    for idx in range(2):  # Run for first 2 records as an example
        print(f"\nProcessing record {idx + 1} with {technique.upper()} prompting on MINERVA...\n")
        prompt = task.get_input(idx)  # Generate input from the task
        minerva_output = get_model_output(prompt, model_name='minerva')  # Use Minerva to get output
        print(f"Minerva output for record {idx + 1}: {minerva_output[0]}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()} bytes")
