import argparse
import torch
from tot.methods.bfs import solve  # Keep this as it is
from tot.models.model_utils import get_model_output  # Import get_model_output
from tot.prompts.game24 import standard_prompt, cot_prompt, propose_prompt  # Import prompts
from tot.tasks.game24 import Game24Task

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Set up the core task (Game of 24)
task = Game24Task()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# Loop over each prompting technique and run the experiment for Minerva
print(f"\nTesting with Falcon backend...\n")

for technique in prompting_techniques:
    print(f"\nRunning experiment with {technique.upper()} prompting technique on Falcon...\n")
    # Modify the arguments for the current technique and backend
    args = argparse.Namespace(
        backend='falcon',
        temperature=0.3,
        task='game24',
        naive_run=False, 
        prompt_sample=None if technique == 'io' else technique,
        method_generate='propose',
        method_evaluate='value', 
        method_select='greedy',
        n_generate_sample=1,
        n_evaluate_sample=3,
        n_select_sample=5
    )
    
    
    # Run the experiment for a few records
    for idx in range(1):  # Run for first 2 records as an example
        print(f"\nProcessing record {idx + 1} with {technique.upper()} prompting on Falcon...\n")
        
        # Retrieve the appropriate prompt template based on the technique
        if technique == 'io':
            prompt_template = standard_prompt
        elif technique == 'cot':
            prompt_template = cot_prompt
        elif technique == 'tot':
            prompt_template = propose_prompt

        # Get the input numbers from the task and format the prompt
        input_numbers = " ".join(map(str, task.get_input(idx)))
        prompt = prompt_template.format(input=input_numbers)
        print("Prompt:", prompt)

        # Use minerva to get output
        try:
            falcon_output = get_model_output(prompt, model_name='falcon', technique=technique)  # Pass technique for accurate response extraction
            print(f"Falcon output for record {idx + 1}: {falcon_output}")
        except Exception as e:
            print(f"Error generating output for record {idx + 1}: {e}")
        
        # # Print GPU memory usage
        # if torch.cuda.is_available():
        #     allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
        #     reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # Convert to GB

  
