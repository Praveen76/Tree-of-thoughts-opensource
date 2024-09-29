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
falcon_model = None
falcon_tokenizer = None
mistral_model = None
mistral_tokenizer = None
completion_tokens = 0
prompt_tokens = 0

claude_client = anthropic.Client(api_key=CLAUDE_API_KEY)

def get_model_output(prompt, model_name, technique, n=1, stop=None):
    # Handling for Llama3
    if model_name == "llama3":
        # Load Llama3 model and tokenizer if not already loaded
        global llama3_model, llama3_tokenizer
        if llama3_model is None or llama3_tokenizer is None:
            print("Loading Llama3 model and tokenizer...")
            llama3_tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", 
                token=HF_READ_API_KEY
            )
            llama3_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", 
                torch_dtype=torch.float16, 
                device_map="auto", 
                token=HF_READ_API_KEY
            )
            llama3_model.eval()

        # Check if pad_token exists, if not, add [PAD] as a new padding token
            if llama3_tokenizer.pad_token is None:
                llama3_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add [PAD] as the padding token
                llama3_model.resize_token_embeddings(len(llama3_tokenizer))  # Resize embeddings to accommodate new token
        
        max_length = 512  
        # Tokenize the input with padding and attention mask
        inputs = llama3_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Generate the output, passing the attention mask and pad_token_id
        with torch.no_grad():
            outputs = llama3_model.generate(
                inputs['input_ids'].to("cuda"), 
                attention_mask=inputs['attention_mask'].to("cuda"),  # Pass the attention mask
                max_length=512, 
                pad_token_id=llama3_tokenizer.pad_token_id  # Ensure pad_token_id is explicitly set
            )
        
        generated_text = llama3_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()

    elif model_name == "mistral":
        # Load Mistral model and tokenizer if not already loaded
        global mistral_model, mistral_tokenizer
        if mistral_model is None or mistral_tokenizer is None:
            print("Loading Mistral model and tokenizer...")
            mistral_tokenizer = AutoTokenizer.from_pretrained(
                 "mistralai/Mistral-7B-v0.1", 
                token=HF_READ_API_KEY
            )
            print("Loading Mistral Model")
            mistral_model = AutoModelForCausalLM.from_pretrained(
                 "mistralai/Mistral-7B-v0.1", 
                torch_dtype=torch.float16, 
                device_map="auto", 
                token=HF_READ_API_KEY
            )
            mistral_model.eval()

        # Check if pad_token exists, if not, add [PAD] as a new padding token
            if mistral_tokenizer.pad_token is None:
                mistral_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add [PAD] as the padding token
                mistral_model.resize_token_embeddings(len(mistral_tokenizer))  # Resize embeddings to accommodate new token
        
        max_length = 512  
        # Tokenize the input with padding and attention mask
        inputs = mistral_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Generate the output, passing the attention mask and pad_token_id
        with torch.no_grad():
            outputs = mistral_model.generate(
                inputs['input_ids'].to("cuda"), 
                attention_mask=inputs['attention_mask'].to("cuda"),  # Pass the attention mask
                max_length=512, 
                pad_token_id=mistral_tokenizer.pad_token_id  # Ensure pad_token_id is explicitly set
            )
        
        generated_text = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()

    elif model_name == "falcon":
        # Load Falcon model and tokenizer if not already loaded
        global falcon_model, falcon_tokenizer
        if falcon_model is None or falcon_tokenizer is None:
            print("Loading Falcon model and tokenizer...")
            falcon_tokenizer = AutoTokenizer.from_pretrained(
                "tiiuae/falcon-7b-instruct", 
                token=HF_READ_API_KEY
            )
            print("Loading Falcon Model")
            falcon_model = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b-instruct", 
                torch_dtype=torch.float16, 
                device_map="auto", 
                token=HF_READ_API_KEY
            )
            falcon_model.eval()

        # Check if pad_token exists, if not, add [PAD] as a new padding token
            if falcon_tokenizer.pad_token is None:
                falcon_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add [PAD] as the padding token
                falcon_model.resize_token_embeddings(len(falcon_tokenizer))  # Resize embeddings to accommodate new token
        
        max_length = 512  
        # Tokenize the input with padding and attention mask
        inputs = falcon_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Generate the output, passing the attention mask and pad_token_id
        with torch.no_grad():
            outputs = falcon_model.generate(
                inputs['input_ids'].to("cuda"), 
                attention_mask=inputs['attention_mask'].to("cuda"),  # Pass the attention mask
                max_length=512, 
                pad_token_id=falcon_tokenizer.pad_token_id  # Ensure pad_token_id is explicitly set
            )
        
        generated_text = falcon_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()

    elif model_name == "minerva":
        global minerva_model, minerva_tokenizer

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
        # Set pad_token_id if not already set
        if minerva_tokenizer.pad_token_id is None:
            minerva_tokenizer.pad_token_id = minerva_tokenizer.eos_token_id
        
        max_length = 512 
        device = minerva_model.device
        inputs = minerva_tokenizer(prompt, return_tensors="pt").to(device)

        # Generate outputs with pad_token_id
        with torch.no_grad():
            outputs = minerva_model.generate(
                **inputs,
                max_length=max_length,  # Increased from 150 to 200
                do_sample=True,
                pad_token_id=minerva_tokenizer.pad_token_id
            )

        generated_text = minerva_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the response after the last occurrence of 'Answer:'
        if 'Answer:' in generated_text[0]:
            response = generated_text[0].split('Answer:')[-1].strip()
        else:
            response = generated_text[0].strip()

        return response

    # Handling for Claude-3
    elif model_name == "claude3-opus":
        try:
            response = claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_text = ''.join([block.text for block in response.content])
            return generated_text.strip()
        except Exception as e:
            return f"Error generating output with Claude: {e}"
    
    # Handling for GPT-4
    elif model_name == "gpt4":
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model="gpt-4", temperature=0.3, max_tokens=150, n=n, stop=stop)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

# GPT-4 logic from models.py
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.3, max_tokens=150, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.3, max_tokens=150, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        completion_tokens += res["usage"]["completion_tokens"]  # This line might fail if res["usage"] is None.

        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        if "usage" in res:
            completion_tokens += res["usage"].get("completion_tokens", 0)
            prompt_tokens += res["usage"].get("prompt_tokens", 0)

    return outputs
