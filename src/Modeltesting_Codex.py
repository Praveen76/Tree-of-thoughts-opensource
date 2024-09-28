import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import openai
from tot.models.model_utils import get_model_output  # Import the function

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Define backends to test (Codex)
backends = ['codex']
# Set up the core task (Game of 24 in this case)
task = Game24Task()

# Loop over each backend and prompting technique and run the experiment
for backend in backends:
    print(f"\nTesting with {backend.upper()} backend...\n")
    
    for technique in prompting_techniques:
        print(f"\nRunning experiment with {technique.upper()} prompting technique on {backend.upper()}...\n")
        
        # Modify the arguments for the current technique and backend
        if technique == 'io':
            # Simple Input-Output prompting (direct)
            args = argparse.Namespace(backend=backend, temperature=0.7, task='game24', naive_run=False, 
                                      prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
        elif technique == 'cot':
            # Chain-of-Thought (step-by-step reasoning)
            args = argparse.Namespace(backend=backend, temperature=0.7, task='game24', naive_run=False, 
                                      prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
        elif technique == 'tot':
            # Tree-of-Thoughts (exploring multiple branches of reasoning)
            args = argparse.Namespace(backend=backend, temperature=0.7, task='game24', naive_run=False, 
                                      prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

        # Run the experiment for a few records (adjust idx for more samples)
        for idx in range(2):  # Run for first 2 records as an example
            print(f"\nProcessing record {idx + 1} with {technique.upper()} prompting on {backend.upper()}...\n")
            
            # Use Codex to generate the output
            prompt = task.get_input(idx)  # Generate input from the task
            codex_output = get_model_output(prompt, model_name='codex')  # Use Codex to get output
            
            # You may want to store or use `codex_output` in your evaluation here
            print(f"Codex output for record {idx + 1}: {codex_output[0]}")
