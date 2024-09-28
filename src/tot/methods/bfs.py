import itertools
import numpy as np
from functools import partial
from tot.models.model_utils import get_model_output

# Get the value based on the backend (GPT or Falcon)
def get_value(task, x, y, n_evaluate_sample, model_name, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    
    value_outputs = get_model_output(value_prompt, model_name=model_name, n=n_evaluate_sample)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    
    if cache_value:
        task.value_cache[value_prompt] = value
    
    return value

# Get values for multiple outputs
def get_values(task, x, ys, n_evaluate_sample, model_name, cache_value=True):
    values = []
    local_value_cache = {}
    
    for y in ys:  # for each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, model_name=model_name, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    
    return values

# Get votes based on the current task
def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = get_model_output(vote_prompt, model_name="gpt-4", n=n_evaluate_sample)  # Currently fixed to GPT
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

# Get proposals based on the backend (GPT or Falcon)
def get_proposals(task, x, y, model_name): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = get_model_output(propose_prompt, model_name=model_name, n=1)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

# Generate samples based on the backend (GPT or Falcon)
def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, model_name):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')

    samples = get_model_output(prompt, model_name=model_name, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

# Solve function
def solve(args, task, idx, to_print=True):
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []

    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step], model_name=args.backend) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, model_name=args.backend) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample, model_name=args.backend)
        
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

# Naive solver
def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None, model_name=args.backend)
    return ys, {}
