"""
Wordle dataset and reward function
"""

import torch


def get_wordle_dataset(tokenizer, split="train"):
    """
    Load wordle dataset from path
    """
    from datasets import load_dataset

    dataset_name = "predibase/wordle-grpo"
    dataset = load_dataset(dataset_name, split=split)

    def parse_qwen_to_target_model_format(x):
        messages = parse_qwen_to_messages(x["prompt"])
        x["input"] = messages_to_model_format(messages, tokenizer)
        return x

    # add new column to dataset
    dataset = dataset.map(parse_qwen_to_target_model_format)

    return dataset


def parse_qwen_to_messages(qwen_prompt):
    """
    Parse a Qwen-formatted chat prompt into a list of messages.
    
    Args:
        qwen_prompt (str): The Qwen-formatted prompt with <|im_start|> and <|im_end|> tags
    
    Returns:
        list: List of message dictionaries with 'role' and 'content' keys
    """
    messages = []
    
    # Split the prompt by <|im_start|> tags
    parts = qwen_prompt.split('<|im_start|>')
    
    for part in parts[1:]:  # Skip the first empty part
        if not part.strip():
            continue
            
        # Check if this part has a closing tag
        if '<|im_end|>' in part:
            # Complete message
            content = part.split('<|im_end|>')[0].strip()
        else:
            # Incomplete message (open-ended, missing <|im_end|>)
            content = part.strip()
        
        # Extract role and content
        lines = content.split('\n', 1)
        if len(lines) >= 1:
            role = lines[0].strip()
            message_content = lines[1].strip() if len(lines) > 1 else ""
            
            # Handle valid roles
            if role in ['system', 'user', 'assistant']:
                messages.append({"role": role, "content": message_content})
    
    return messages


def messages_to_model_format(messages, tokenizer, add_generation_prompt=True):
    """
    Convert a list of messages to target model format.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys
        target_model_name (str): Target model name
        add_generation_prompt (bool): Whether to add generation prompt for incomplete conversations
    
    Returns:
        str: Formatted prompt for the target model
    """
    try:        
        # Handle model-specific message processing
        processed_messages = messages.copy()
        
        if not processed_messages:
            return None
            
        # Apply the target model's chat template
        formatted_prompt = tokenizer.apply_chat_template(
            processed_messages, 
            tokenize=False, 
            continue_final_message=True,
        )
        
        return formatted_prompt
        
    except Exception as e:
        print(f"Error converting messages to model format: {e}")
        return None



def reward_wordle(y_true: str, y_pred: str) -> float:
    """
    Reward function to calculate reward for wordle guess.
    Heuristic:
    if y_pred is invalid (!=5 or non-alpha) : 0
    for each position correct : 0.1 x n
    for each alpha correct in wrong position: 0.05 x n
    if extact match : 1
    """
    y_true = y_true.strip().upper()
    y_pred = y_pred.strip().upper()
    # Check validity
    if len(y_pred) != 5 or not y_pred.isalpha():
        return 0.0
    if y_pred == y_true:
        return 1.0
    reward = 0.0
    # Count correct positions
    for i in range(5):
        if y_pred[i] == y_true[i]:
            reward += 0.1
    # Count correct letters in wrong positions
    # To avoid double-counting, mark matched positions
    true_counts = {}
    pred_counts = {}
    for i in range(5):
        if y_pred[i] != y_true[i]:
            true_counts[y_true[i]] = true_counts.get(y_true[i], 0) + 1
            pred_counts[y_pred[i]] = pred_counts.get(y_pred[i], 0) + 1
    for letter in pred_counts:
        if letter in true_counts:
            reward += 0.05 * min(pred_counts[letter], true_counts[letter])
    return reward


def extract_guess_from_text(text):
    """
    Extracts the guess from between <guess> and </guess> tags in the input text.
    There should be only one pair of <guess>...</guess> tags.
    Returns the guess as a string, stripped of whitespace.
    Raises ValueError if the tags are missing or if there are multiple pairs.
    """
    import re
    matches = re.findall(r'<guess>(.*?)</guess>', text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one <guess>...</guess> tag, found {len(matches)}.")
    return matches[0].strip()