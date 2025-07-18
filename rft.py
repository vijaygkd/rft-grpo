"""
RFT - Reinforcement Fine-Tuning main file
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

from grpo import grpo_loss_fn
from wordle import get_wordle_dataset, get_wordle_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def train_grpo():
    """
    Train GRPO as per DeepSeek-Math / GRPO paper (https://arxiv.org/abs/2402.03300)
    """
    # 1. Load dataset - Wordle
    dataset = get_wordle_dataset(tokenizer)
    print(f"Dataset loaded with {len(dataset)} examples")

    # 2. Load model - Gemma-3-1b-it
    model_name = "google/gemma-3-1b-it"
    print(f"Init models: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ref_model = init_lora_model(model_name).requires_grad_(False).to(device)    # frozen
    old_model = init_lora_model(model_name).requires_grad_(False).to(device)    # frozen
    curr_model = init_lora_model(model_name).requires_grad_(True).to(device)    # trainable
    assert_model_same(curr_model, old_model)

    print("Ref model info:")
    print_model_info(ref_model)
    print("Old model info:")
    print_model_info(old_model)
    print("Curr model info:")
    print_model_info(curr_model)

    # 3. Training loop
    num_epochs = 1
    batch_size = 1          # only one sample at a time for now TODO: batching
    num_generations = 4     # number of generations per sample
    num_rl_steps = 4        # number of RL steps per mini-batch
    max_seq_len = 512       # max sequence length

    optimizer = torch.optim.AdamW(curr_model.parameters(), lr=1e-4)
    # TODO: add learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        print("-"*100)
        print(f"Epoch {epoch+1}/{num_epochs}")
        dataset = dataset.shuffle(seed=epoch)
        dataset = dataset.select(range(3))

        # Anchor KL ref model to current model policy
        # - Allows Trust Region to move slowly allowing exploration of new policies (similar to PPO)
        copy_model_weights(source_model=curr_model, target_model=ref_model)
        assert_model_same(curr_model, ref_model)    # TODO: remove later

        for samples in dataset.iter(batch_size=batch_size):
            print("."*30)
            # 1. ACTION - Sample generations
            # Update old model policy to current model policy at each mini-batch
            copy_model_weights(source_model=curr_model, target_model=old_model)
            assert_model_same(curr_model, old_model)
            # Sample generations from old model policy once and do multiple RL steps
            batch_input_ids, batch_output_masks, batch_output_texts = generate_batch(
                model=old_model,
                tokenizer=tokenizer, 
                sample_inputs=samples['input'], 
                num_generations=num_generations, 
                max_seq_len=max_seq_len
            )
            secret_word = samples['secret'][0]
            
            print(f"SECRET: {secret_word}")
            for i, text in enumerate(batch_output_texts):
                print(f"Generation {i}: {text}")

            # 2. REWARD - Calculate rewards and advantage using group relative preference
            # - Calculate rewards for each generation
            batch_rewards = get_wordle_rewards(batch_output_texts, secret_word)
            batch_rewards = torch.tensor(batch_rewards, device=device)
            print(f"Batch rewards: {batch_rewards}")
            
            # 3. GRPO - RL loop on same outputs
            for k in range(num_rl_steps):
                print("."*30)
                print(f"RL step {k+1}/{num_rl_steps}")
                # zero grad
                optimizer.zero_grad()
                # Calculate GRPO loss
                grpo_loss = grpo_loss_fn(
                    curr_model=curr_model, 
                    old_model=old_model, 
                    ref_model=ref_model, 
                    seq_ids=batch_input_ids, 
                    output_masks=batch_output_masks, 
                    rewards=batch_rewards,
                    pad_token_id=tokenizer.pad_token_id
                )
                print(f"GRPO loss: {grpo_loss}")
                # gradient clipping
                print(f"Model grad norm before clipping: {model_grad_norm(curr_model)}")
                torch.nn.utils.clip_grad_norm_(curr_model.parameters(), max_norm=1.0)
                print(f"Model grad norm after clipping: {model_grad_norm(curr_model)}")
                # BACKPROP - Update curr model with RL
                grpo_loss.backward()
                optimizer.step()
                # TODO: add learning rate scheduler
                # lr_scheduler.step()    


            # TODO:
            # - add logs - print loss, reward, plot graph
            # - add evaluation on val set
            # - add checkpointing
            # - add early stopping
            # - add learning rate scheduler
            # - add mixed precision training
            # - add gradient accumulation

    print("Training complete")


def model_grad_norm(model: torch.nn.Module) -> float:
    """
    Calculate the gradient norm of the model
    """
    # check if all grads are None
    if all(p.grad is None for p in model.parameters()):
        return 0.0
    else:
        return torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))


def generate_batch(model, tokenizer, sample_inputs, num_generations, max_seq_len):
    """
    Generate batch of generations from "old" model

    Note: only works for single input sample at a time
    """
    # Extract input texts from samples
    all_sequences = []
    all_output_masks = []
    all_output_texts = []
    
    # Process each input sample
    for sample_input in sample_inputs:
        # Tokenize current input
        input_pt = tokenizer(sample_input, return_tensors="pt").to(device)
        input_len = input_pt.input_ids.shape[1]
        
        # Generate sequences for this input
        with torch.no_grad():
            sample_gen = model.generate(
                **input_pt,
                num_return_sequences=num_generations,
                max_new_tokens=max_seq_len,
                temperature=1,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Create output masks for this sample
        output_masks = torch.ones(sample_gen.sequences.shape).to(device)
        output_masks[:, :input_len] = 0  # mask input tokens
        
        # Handle padding tokens
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            padding_mask = (sample_gen.sequences == pad_token_id)
            output_masks[padding_mask] = 0
        
        all_sequences.append(sample_gen.sequences)
        all_output_masks.append(output_masks)
        all_output_texts.extend(
            tokenizer.batch_decode(
                sample_gen.sequences[:, input_len:], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        )

    # Concatenate all sequences and masks along batch dimension
    batch_input_ids = torch.cat(all_sequences, dim=0)
    batch_output_masks = torch.cat(all_output_masks, dim=0)
    batch_output_texts = all_output_texts
    
    return batch_input_ids, batch_output_masks, batch_output_texts


def print_model_info(model):
    """
    Print model info
    """
    #num trainable parameters
    print(f"No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    #no. of non-trainable parameters
    print(f"No. of non-trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")


def load_base_model(model_name) -> AutoModelForCausalLM:
    """
    Load base model and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    return model


def init_lora_model(model_name: str) -> PeftModel:
    """
    Initialize LoRA model from base model
    """
    # load base model
    model = load_base_model(model_name)
    # initialize LoRA config
    lora_config = LoraConfig(
        r=8,    # rank of LoRA matrix
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        init_lora_weights=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # initialize LoRA model
    model = get_peft_model(model, lora_config)
    return model


def copy_model_weights(source_model: nn.Module, target_model: nn.Module):
    """
    Copy source  model weights to target model in place
    """
    for (name1, param1), (name2, param2) in zip(source_model.named_parameters(), target_model.named_parameters()):
        assert name1 == name2, "Model parameters are not the same"
        param2.data.copy_(param1.data)


def assert_model_same(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Model parameters are not the same: {name1} != {name2}"
        assert param1.dtype == param2.dtype, f"Model parameters are not the same for {name1}: {param1.dtype} != {param2.dtype}"
        assert torch.allclose(param1.data, param2.data), f"Model parameters are not the same for {name1}: {param1.data} != {param2.data}"
    print("Model parameters are the same")



# 1. Load model - Llama7b model

# 2. Load dataset - Wordle

# 3. Build reward function

# 4. GRPO implementation - loss

# 5. Training loop

# 6. Bechmarking & Evaluation





if __name__ == "__main__":
    # Uncomment to test the batch input function
    # test_get_batch_input()
    
    # Run full training
    train_grpo()