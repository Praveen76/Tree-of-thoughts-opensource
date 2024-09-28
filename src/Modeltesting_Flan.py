import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Define backends to test (Flan-T5)
backends = ['flan-t5']  # Focusing on Flan-T5

# Set up the core task (Game of 24 in this case)
task = Game24Task()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# Load the Flan-T5 model once and reuse it
model_name = "google/flan-t5-large"  # You can also use "flan-t5-base" if you want something lighter
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to get model output using Flan-T5
def get_flan_output(prompt, n=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,  # Limit token generation
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_output

# Loop over each backend and prompting technique and run the experiment
for backend in backends:
    print(f"\nTesting with {backend.upper()} backend...\n")
    
    for technique in prompting_techniques:
        print(f"\nRunning experiment with {technique.upper()} prompting technique on {backend.upper()}...\n")
        
        # Modify the arguments for the current technique and backend
        if technique == 'io':
            # Simple Input-Output prompting (direct)
            args = argparse.Namespace(backend=backend, temperature=0.5, task='game24', naive_run=False, 
                                      prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
        elif technique == 'cot':
            # Chain-of-Thought (step-by-step reasoning)
            args = argparse.Namespace(backend=backend, temperature=0.5, task='game24', naive_run=False, 
                                      prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
        elif technique == 'tot':
            # Tree-of-Thoughts (exploring multiple branches of reasoning)
            args = argparse.Namespace(backend=backend, temperature=0.5, task='game24', naive_run=False, 
                                      prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

        # Run the experiment for a few records (adjust idx for more samples)
        for idx in range(2):  # Run for first 2 records as an example
            print(f"\nProcessing record {idx + 1} with {technique.upper()} prompting on {backend.upper()}...\n")
            
            # Use Flan-T5 to generate the output
            prompt = task.get_input(idx)  # Generate input from the task
            flan_output = get_flan_output(prompt)  # Use Flan-T5 to get output
            
            # Print the output
            print(f"Flan-T5 output for record {idx + 1}: {flan_output[0]}")
            
            # GPU memory information
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()} bytes")
