"""
RFT - Reinforcement Fine-Tuning main file
"""
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from grpo import grpo_loss_fn
from wordle import get_wordle_dataset, get_wordle_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def train_rft():
    """
    Train RFT
    """
    # 1. Load model - Gemma-3-1b-it
    model_name = "google/gemma-3-1b-it"
    print(f"Loading model: {model_name}")
    model, tokenizer = load_base_model(model_name)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.to(device)
    print("Ref model info:")
    print_model_info(ref_model)
    prev_model = copy.deepcopy(model)
    prev_model.eval()
    prev_model.requires_grad_(False)
    prev_model.to(device)


    # 2. Load LoRA model
    model = load_lora_model(model)
    model.to(device)
    print("LoRA model info:")
    print_model_info(model)

    # 2. Load dataset - Wordle
    # TODO - dataloader to shuffle and batch
    dataset = get_wordle_dataset(tokenizer)
    dataset = dataset.select(range(10))
    print(f"Loaded {len(dataset)} examples")
    

    # 3. Training Parameters
    num_epochs = 10
    num_generations = 4    # number of generations per sample
    # only one sample at a time for now
    num_samples = 1        # number of input samples: batch size = num_samples * num_generations
    max_seq_len = 128      # max sequence length

    # 4. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*100)

        for samples in dataset.iter(batch_size=num_samples):
            # 1. ACTION - Sample generations from model
            batch_input_ids, batch_output_masks, batch_output_texts = generate_batch(
                model=model, 
                tokenizer=tokenizer, 
                sample_inputs=samples['input'], 
                num_generations=num_generations, 
                max_seq_len=max_seq_len
            )

            print(f"Batch output texts: {batch_output_texts}")
            print(f"secret: {samples['secret'][0]}")

            # 2. REWARD - Calculate rewards
            secret_word = samples['secret'][0]
            batch_rewards = get_wordle_rewards(batch_output_texts, secret_word)
            batch_rewards = torch.tensor(batch_rewards, device=device)
            print(f"Batch rewards: {batch_rewards}")
            
            # 3. GRPO - Calculate GRPO loss
            grpo_loss = grpo_loss_fn(
                curr_model=model, 
                old_model=prev_model, 
                ref_model=ref_model, 
                seq_ids=batch_input_ids, 
                output_masks=batch_output_masks, 
                rewards=batch_rewards
            )
            print(f"GRPO loss: {grpo_loss}")
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 4. BACKPROP - Update model
            # update prev model before backprop for next iteration
            prev_model = copy.deepcopy(model)
            prev_model.requires_grad_(False)
            # update curr model
            optimizer.zero_grad()
            grpo_loss.backward()
            optimizer.step()                # TODO : add gradient accumulation


            print("."*30)



            # TODO:
            # - add logs - print loss, reward, plot graph
            # - add evaluation on val set
            # - add checkpointing
            # - add early stopping
            # - add learning rate scheduler
            # - add mixed precision training
            # - add gradient accumulation
 

    return



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



def load_base_model(model_name):
    """
    Load base model
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_lora_model(model):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,    # rank of LoRA matrix
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        init_lora_weights=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Load LoRA weights
    model = get_peft_model(model, lora_config)

    return model



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
    train_rft()