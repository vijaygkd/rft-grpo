"""
RFT - Reinforcement Fine-Tuning main file
"""
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from wordle import get_wordle_dataset, reward_wordle
from grpo import get_model_log_prob, grpo_loss


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

    model = load_lora_model(model)
    model.to(device)
    print("LoRA model info:")
    print_model_info(model)

    # 2. Load dataset - Wordle
    dataset = get_wordle_dataset(tokenizer)
    print(f"Loaded {len(dataset)} examples")

    # 3. Training Parameters
    num_epochs = 10
    num_generations = 2    # number of generations per sample
    num_samples = 1        # number of input samples: batch size = num_samples * num_generations
    max_seq_len = 128      # max sequence length

    # 4. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*100)

        for samples in dataset.iter(batch_size=num_samples):
            batch_input, batch_output_masks = get_batch_input(samples['input'], model, tokenizer, num_generations, max_seq_len)

            # TODO train
            continue


def get_batch_input(sample_inputs, model, tokenizer, num_generations, max_seq_len):
    """
    Get batch input for multiple samples - basic version
    """
    # Extract input texts from samples
    all_sequences = []
    all_output_masks = []
    
    # Process each input sample
    for sample_input in sample_inputs:
        # Tokenize current input
        input_pt = tokenizer(sample_input, return_tensors="pt").to(device)
        input_len = input_pt.input_ids.shape[1]
        
        # Generate sequences for this input
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
    
    # Concatenate all sequences and masks along batch dimension
    batch_sequences = torch.cat(all_sequences, dim=0)
    batch_output_masks = torch.cat(all_output_masks, dim=0)
    
    print(f"Batch shape: {batch_sequences.shape}, pad_token_id: {tokenizer.pad_token_id}")
    
    return batch_sequences, batch_output_masks


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