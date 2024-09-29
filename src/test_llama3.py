# test_llama3.py

from tot.models.model_utils import get_model_output

test_prompt_io = '''Use the given numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 1 1 4 6
Answer:'''

test_prompt_cot = '''Use the given numbers and basic arithmetic operations (+ - * /) to obtain 24. Show your reasoning step by step.
Input: 4 4 6 8
Steps:
1. 4 + 8 = 12 (remaining numbers: 4, 6, 12)
2. 6 - 4 = 2 (remaining numbers: 2, 12)
3. 2 * 12 = 24 (remaining numbers: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 1 4 8 8
Steps:
1. 8 / 4 = 2 (remaining numbers: 1, 2, 8)
2. 1 + 2 = 3 (remaining numbers: 3, 8)
3. 3 * 8 = 24 (remaining numbers: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 1 1 4 6
Steps:'''

test_prompt_tot = '''Consider all possible next steps using the given numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 2 8 8 14
Possible next steps:
- 2 + 8 = 10 (remaining numbers: 8, 10, 14)
- 8 / 2 = 4 (remaining numbers: 4, 8, 14)
- 14 - 8 = 6 (remaining numbers: 2, 6, 8)
Input: 1 1 4 6
Possible next steps:'''

# Test IO Prompt
output_io = get_model_output(test_prompt_io, model_name='llama3', technique='io')
print("IO Prompt Output:", output_io)

# Test COT Prompt
output_cot = get_model_output(test_prompt_cot, model_name='llama3', technique='cot')
print("COT Prompt Output:", output_cot)

# Test TOT Prompt
output_tot = get_model_output(test_prompt_tot, model_name='llama3', technique='tot')
print("TOT Prompt Output:", output_tot)
