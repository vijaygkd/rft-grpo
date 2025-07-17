"""
GRPO loss function implementation
Reference: Deep-Seek-R1 paper
"""

import torch


def get_model_log_prob(model, seq_ids, output_masks, pad_token_id=None):
    """
    Calculate model sequence log probability of OUTPUT given INPUT: P(output | input)
    
    Args:
        model: Model to use for inference
        seq_ids: Tensor of shape (B, seq_len) containing token ids of generated sequence belonging to same input
        output_masks: Tensor of shape (B, seq_len) containing mask for ouptput positions. (input and padding positions in seq_ids are set to 0)
        pad_token_id: Token id for padding token. If None, will use no attention mask.
    Returns:
        Tensor of shape (B,) containing log probabilities for each outputsequence in batch
        
    Note:
        Uses model.forward() to get logits for sequence, then uses log_softmax to get log probabilities.
        Using model.forward() helps track gradients of output tensor for backprop.
    """
    # opmask: 0 0 0 0 0 1 1 1 1 1 1     (masks of output positions, ignore padding tokens)
    # input : i n p u t o u t p u t     (model input seq)
    # output: n p u t o u t p u t -     (get logit for output tokens, ie. model probablity)
    # masks:  0 0 0 0 1 1 1 1 1 1 0     (left shift by 1 to get logits for output tokens)
    
    assert seq_ids.shape == output_masks.shape, "seq_ids and output_masks must have same shape"
    attention_mask = None
    if pad_token_id:
        attention_mask = torch.where(seq_ids == pad_token_id, 0, 1).type(torch.int64)
    
    # forward pass
    output = model(
        input_ids=seq_ids,
        attention_mask=attention_mask,
    )
    logits = output.logits
    token_log_prob = torch.log_softmax(logits, dim=-1)              # (B, seq_len, V)  -> log_prob for each token in seq
    output_indices = torch.roll(seq_ids, shifts=-1, dims=-1)        # (B, seq_len) -> left shift by 1 to get indices for output tokens
    token_log_prob = token_log_prob.gather(
        index=output_indices.unsqueeze(-1),                         # (B, seq_len, 1)  -> (B, seq_len, 1) 
        dim=-1
    )                                                               # (B, seq_len, 1)  -> log_prob for seq tokens
    token_log_prob = token_log_prob.squeeze()                                   # (B, seq_len)
    loss_mask = torch.roll(output_masks, shifts=-1, dims=-1)            # left shift by 1
    loss_mask[:, -1] = 0                                                # pad last token with 0 as it is not output token
    output_token_log_prob = token_log_prob * loss_mask                  # (B, seq_len)  -> log_prob for output tokens     
    output_log_prob = output_token_log_prob.sum(dim=-1)             # (B, 1)  -> sum of log_prob for output tokens
    return output_log_prob


def get_group_relative_reward_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calculate reward advantage of sequence with "Group Relevative" reward function.
    
    Args:
        rewards: Tensor of shape (B,) containing reward values for generated sequences belonging to same input
    
    Returns:
        Tensor of shape (B,) containing reward advantage based on group relative reward function.
    """
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    advantages = (rewards - mean_reward) / (std_reward + eps)
    return advantages


def get_kl_divergence(curr_log_prob, ref_log_prob):
    """
    Calculate KL divergence between model and ref_model for sequence given input.

    Args:
        curr_log_prob: Tensor of shape (B,) containing log probabilities for current model
        ref_log_prob: Tensor of shape (B,) containing log probabilities for reference model

    Returns:
        Tensor of shape (B,) containing KL divergence for each outputsequence in batch
    """
    kl_div = torch.exp(ref_log_prob - curr_log_prob) - (ref_log_prob - curr_log_prob) - 1
    return kl_div
    

def grpo_loss_fn(curr_model, old_model, ref_model, seq_ids, output_masks, rewards, pad_token_id=None, ep=0.2, beta=0.1):
    """
    GRPO loss function

    Args:
        curr_model: current policy model - trained model weights
        old_model: old policy model - old model weights from previous iteration
        ref_model: Reference model base model - pretrained model weights
        seq_ids: Tensor of shape (B, seq_len) containing token ids of generated sequence belonging to same input
        output_masks: Tensor of shape (B, seq_len) containing mask for ouptput positions. (input and padding positions in seq_ids are set to 0)
        rewards: Tensor of shape (B,) containing reward values for generated sequences belonging to same input
        ep: clipping threshold for reward
        beta: KL divergence regularization parameter

    Returns:
        Tensor of shape (1,) containing GRPO loss for the batch
    """
    with torch.no_grad():
        # no grad for old and ref models
        ref_log_prob = get_model_log_prob(ref_model, seq_ids, output_masks, pad_token_id)
        old_log_prob = get_model_log_prob(old_model, seq_ids, output_masks, pad_token_id)
    # with grad for curr model -- for backprop
    curr_log_prob = get_model_log_prob(curr_model, seq_ids, output_masks, pad_token_id)   # (B,)

    print(f"curr_log_prob: {curr_log_prob}")
    print(f"curr_prob: {torch.exp(curr_log_prob)}")
    print(f"old_log_prob: {old_log_prob}")
    print(f"old_prob: {torch.exp(old_log_prob)}")
    print(f"ref_log_prob: {ref_log_prob}")
    print(f"ref_prob: {torch.exp(ref_log_prob)}")

    # policy objective
    adv = get_group_relative_reward_advantage(rewards)                     
    ratio = torch.exp(curr_log_prob - old_log_prob)
    print(f"ratio: {ratio}")
    unclipped = ratio * adv                                                
    clipped = torch.clamp(ratio, 1 - ep, 1 + ep) * adv                     
    policy_obj = torch.min(unclipped, clipped)                             
    # kl divergence
    kl_div = get_kl_divergence(curr_log_prob, ref_log_prob)           
    print(f"kl_div: {kl_div}")
    # grpo loss
    grpo_loss = policy_obj - (beta * kl_div)                                # (B,)
    print(f"grpo_loss for batch: {grpo_loss}")
    grpo_loss = - grpo_loss.mean()                                          # (1,)
    return grpo_loss
