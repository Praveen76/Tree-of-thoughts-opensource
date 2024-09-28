import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the prompting techniques you want to test
prompting_techniques = ['io', 'cot', 'tot']

# Define backends to test (Falcon-7B)
backends = ['falcon-7b']  # Focusing on Falcon-7B

# Set up the core task (Game of 24 in this case)
task = Game24Task()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# Load the Falcon model once and reuse it
model_name = "tiiuae/falcon-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to get model output using Falcon-7B
def get_falcon_output(prompt, n=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=1000,  # Adjust token generation based on requirements
        do_sample=True,
        temperature=0.7,  # Increased temperature for more diversity
        top_p=0.9,  # Use nucleus sampling
        repetition_penalty=1.2,  # Penalizes repetition
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=inputs['attention_mask']
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
            args = argparse.Namespace(backend=backend, temperature=0.7, task='game24', naive_run=False, 
                                      prompt_sample=None, method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
        elif technique == 'cot':
            args = argparse.Namespace(backend=backend, temperature=0.7, task='game24', naive_run=False, 
                                      prompt_sample='cot', method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
        elif technique == 'tot':
            args = argparse.Namespace(backend=backend, temperature=0.7, task='game24', naive_run=False, 
                                      prompt_sample='tot', method_generate='propose', method_evaluate='value', 
                                      method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

        # Run the experiment for a few records
        for idx in range(2):  # Run for first 2 records as an example
            print(f"\nProcessing record {idx + 1} with {technique.upper()} prompting on {backend.upper()}...\n")
            
            # Use Falcon to generate the output
            prompt = f"Solve this 24 game puzzle with the numbers {task.get_input(idx)}. Use each number once with the operators +, -, *, /."
            print(f"Generated prompt for record {idx + 1}: {prompt}")
            
            falcon_output = get_falcon_output(prompt)  # Use Falcon to get output
            
            # You may want to store or use `falcon_output` in your evaluation here
            print(f"Falcon output for record {idx + 1}: {falcon_output[0]}")
            
            # GPU memory information
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()} bytes")

            # Log if the output seems repetitive (early detection of the looping issue)
            if len(set(falcon_output[0].split())) == 1:
                print(f"Warning: Repetitive output detected for record {idx + 1} on {backend.upper()}")
