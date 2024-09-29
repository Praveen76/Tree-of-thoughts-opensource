# src/tot/prompts/game24.py

# 5-shot Standard Prompt
standard_prompt = '''Use the given numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
Answer:'''

# 5-shot Chain-of-Thought Prompt
cot_prompt = '''Use the given numbers and basic arithmetic operations (+ - * /) to obtain 24. Show your reasoning step by step.
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
Input: {input}
Steps:'''

# 1-shot Tree-of-Thought Prompt
propose_prompt = '''Consider all possible next steps using the given numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 2 8 8 14
Possible next steps:
- 2 + 8 = 10 (remaining numbers: 8, 10, 14)
- 8 / 2 = 4 (remaining numbers: 4, 8, 14)
- 14 - 8 = 6 (remaining numbers: 2, 6, 8)
Input: {input}
Possible next steps:'''

# Simplified Value Prompt (3 examples)
value_prompt = '''Evaluate whether the given numbers can reach 24 using basic arithmetic operations (+ - * /). Respond with "sure", "likely", or "impossible".
Numbers: 10 14
Evaluation:
10 + 14 = 24
Answer: sure
Numbers: 11 12
Evaluation:
11 + 12 = 23
11 * 12 = 132
Answer: impossible
Numbers: 4 9 11
Evaluation:
4 + 9 + 11 = 24
Answer: sure
Numbers: {input}
Evaluation:'''

# Simplified Value Last Step Prompt (2 examples)
value_last_step_prompt = '''Determine if the provided answer correctly uses each input number exactly once to reach 24 using basic arithmetic operations (+ - * /). Respond with "sure" or "impossible".
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judgment: sure
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judgment: impossible
Input: {input}
Answer: {answer}
Judgment:'''

__all__ = ['standard_prompt', 'cot_prompt', 'propose_prompt', 'value_prompt', 'value_last_step_prompt']
