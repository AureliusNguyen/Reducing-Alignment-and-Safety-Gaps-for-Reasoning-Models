import sys
import json
from .constraint_registry import INSTRUCTION_DICT

def test_instruction_following_strict(
    instruction_id_list,
    response,
    parameters,
    prompt,
):
    """Tests response to see if instructions are followed."""
    is_following_list = []
    
    # Handle edge case where instruction_id_list might be empty
    if not instruction_id_list:
        return [True]

    for index, instruction_id in enumerate(instruction_id_list):
        try:
            instruction_cls = INSTRUCTION_DICT[instruction_id]
        except KeyError:
            print(f"Warning: Instruction ID {instruction_id} not found.")
            is_following_list.append(False)
            continue
            
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs
        if parameters and index < len(parameters) and parameters[index]:
            kwargs = {n: p for n, p in parameters[index].items() if p}
        else:
            kwargs = {}
            
        instruction.build_description(**kwargs)
        args = instruction.get_constraint_args()
        
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
            
        try:
            # Check if response follows the instruction
            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)
        except Exception as e:
            # In RL, we usually return False on crash rather than stopping the training
            print(f"Error checking instruction: {e}")
            is_following_list.append(False)

    return is_following_list

def verify_single_response(response_str, data_dict):
    """
    Args:
        response_str (str): The raw text output from the model.
        data_dict (dict): The dataset entry containing 'constraint_name', 
                          'constraint_args', 'question', and 'answer'.
    Returns:
        dict: Contains 'strict_score', 'loose_score'
    """
    
    # --- 1. Pre-process / Clean the Response (Logic extracted from original loop) ---
    hypothesis = response_str
    
    # Handle list format if applicable
    if isinstance(hypothesis, list):
        hypothesis = hypothesis[0]

    # Clean <think> tags
    if '<think>' in hypothesis:
        hypothesis = hypothesis.split('<think>')[1]
    if '</think>' in hypothesis:
        hypothesis = hypothesis.split('</think>')[1]
        
    # Clean <answer> tags
    if '<answer>' in hypothesis:
        hypothesis = hypothesis.split('<answer>')[1]
    if '</answer>' in hypothesis:
        hypothesis = hypothesis.split('</answer>')[0] 

    # --- 2. Check Instructions ---
    # Ensure keys exist in data_dict
    constraint_names = data_dict.get("constraint_name", [])
    constraint_args = data_dict.get("constraint_args", [])
    question = data_dict.get("question", "")

    is_follow_list = test_instruction_following_strict(
        constraint_names,
        hypothesis,
        constraint_args,
        question,
    )

    if not is_follow_list:
        # If no constraints existed, usually considered pass (1.0)
        strict_score = 1.0
        loose_score = 1.0
    else:
        strict_score = 1.0 if all(is_follow_list) else 0.0
        loose_score = sum(is_follow_list) / len(is_follow_list)


    return {
        "strict_score": strict_score,
        "loose_score": loose_score
    }