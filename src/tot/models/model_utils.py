# src/tot/models/model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
import anthropic
import yaml
import os
from huggingface_hub import login, whoami

# Load API keys from YAML file
file_path = './API_KEYS.yml'
with open(file_path, 'r') as file:
    api_keys = yaml.safe_load(file)

# Extract API keys
openai_key = api_keys['OPEN_AI']['Key']
HF_READ_API_KEY = api_keys['HUGGINGFACE']['HF_READ_API_KEY']
LANGCHAIN_API_KEY = api_keys['LANGCHAIN_API_KEY']['Key']
CLAUDE_API_KEY = api_keys['Anthropic_AI']['key']

# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["HUGGINGFACE_TOKEN"] = HF_READ_API_KEY
os.environ["CLAUDE_API_KEY"] = CLAUDE_API_KEY

# Initialize OpenAI API
openai.api_key = openai_key

# Login to Hugging Face
login(token=HF_READ_API_KEY)

# Retrieve user info
user_info = whoami()
print(user_info)

# Global variables for caching models
llama3_model = None
llama3_tokenizer = None
minerva_model = None
minerva_tokenizer = None

def get_model_output(prompt, model_name, technique, n=1, stop=None):
    if model_name == "llama3":
        global llama3_model
        global llama3_tokenizer
        if llama3_model is None or llama3_tokenizer is None:
            print("Loading Llama3 model and tokenizer...")
            # Replace 'your_llama3_model_identifier' with the actual model path or identifier
            llama3_tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                token=HF_READ_API_KEY  # Updated from use_auth_token=True to token=HF_READ_API_KEY
            )
            llama3_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                torch_dtype=torch.float16,
                device_map="auto",
                token=HF_READ_API_KEY  # Updated similarly
            )
            llama3_model.eval()
            print("Llama3 model and tokenizer loaded.")
        else:
            print("Llama3 model and tokenizer already loaded.")

        device = llama3_model.device
        inputs = llama3_tokenizer(prompt, return_tensors="pt").to(device)

        # Set pad_token_id if not already set
        if llama3_tokenizer.pad_token_id is None:
            llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id

        # Generate outputs with pad_token_id
        with torch.no_grad():
            outputs = llama3_model.generate(
                **inputs,
                max_new_tokens=200,  # Increased from 150 to 200 for better COT and TOT responses
                do_sample=True,
                temperature=0.3,     # Adjusted as per your args
                pad_token_id=llama3_tokenizer.pad_token_id
                # Removed 'attention_mask=inputs['attention_mask']' to avoid duplication
            )

        generated_text = llama3_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the response after the last occurrence of 'Answer:' or 'Possible next steps:'
        if technique == 'io' and 'Answer:' in generated_text[0]:
            response = generated_text[0].split('Answer:')[-1].strip()
        elif technique == 'cot' and 'Answer:' in generated_text[0]:
            response = generated_text[0].split('Answer:')[-1].strip()
        elif technique == 'tot' and 'Possible next steps:' in generated_text[0]:
            response = generated_text[0].split('Possible next steps:')[-1].strip()
        else:
            response = generated_text[0].strip()

        return response

    elif model_name == "minerva":
        global minerva_model
        global minerva_tokenizer
        if minerva_model is None or minerva_tokenizer is None:
            print("Loading Minerva model and tokenizer...")
            minerva_tokenizer = AutoTokenizer.from_pretrained(
                "sapienzanlp/Minerva-3B-base-v1.0"
            )
            minerva_model = AutoModelForCausalLM.from_pretrained(
                "sapienzanlp/Minerva-3B-base-v1.0",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            minerva_model.eval()
            print("Minerva model and tokenizer loaded.")
        else:
            print("Minerva model and tokenizer already loaded.")

        device = minerva_model.device
        inputs = minerva_tokenizer(prompt, return_tensors="pt").to(device)

        # Set pad_token_id if not already set
        if minerva_tokenizer.pad_token_id is None:
            minerva_tokenizer.pad_token_id = minerva_tokenizer.eos_token_id

        # Generate outputs with pad_token_id
        with torch.no_grad():
            outputs = minerva_model.generate(
                **inputs,
                max_new_tokens=200,  # Increased from 150 to 200
                do_sample=True,
                temperature=0.7,
                pad_token_id=minerva_tokenizer.pad_token_id
                # Removed 'attention_mask=inputs['attention_mask']' to avoid duplication
            )

        generated_text = minerva_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the response after the last occurrence of 'Answer:'
        if 'Answer:' in generated_text[0]:
            response = generated_text[0].split('Answer:')[-1].strip()
        else:
            response = generated_text[0].strip()

        return response

    else:
        raise ValueError(f"Model {model_name} is not supported.")
