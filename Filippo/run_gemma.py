import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc

# --- 1. Setup ---
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # Enable optimizations for AMD GPUs if applicable
model_id = "google/gemma-3-1b-it" # Gemma 3 1B Instruction-Tuned
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

print(f"Loading {model_id} on {device}...")

# --- 2. Load Model ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16, 
    device_map=device,
    attn_implementation="sdpa"
)

# Fix padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. Optimized Prompts ---
prompt_1 = """You are a pipeline architect. Fill in the missing [Step 2].
Constraint: Step 2 must bridge the gap between Step 1 and Step 3. No fancy formattinglike bold or italics.

[Step 1]
Name: Raw Data Extraction
Instructions: Connect to SQL database, query Q3 transaction logs, and export raw results to JSON.

[Step 3]
Name: Generate Visual Report
Instructions: Using the finalized dataset, generate a bar chart comparing monthly revenue streams.

[Step 2]
Name:"""

prompt_2 = """You are a pipeline architect. Fill in the missing [Step 2].
Constraint: Step 2 must bridge the gap between Step 1 and Step 3. No fancy formattinglike bold or italics.

[Step 1]
Name: Email Ingestion
Instructions: Retrieve unread emails from the customer support inbox and strip out HTML signatures and headers.

[Step 3]
Name: Ticket Routing System
Instructions: Based on the identified category and urgency level, assign the structured ticket object to the correct support team queue.

[Step 2]
Name:"""

# --- 4. Prepare Inputs ---
# We use the Chat Template because Gemma-1B-IT is trained on it.
# It is much smarter when we use the correct template than raw text.
messages = [{"role": "user", "content": prompt_2}]

encodings = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt", 
    add_generation_prompt=True,
    return_dict=True 
).to(device)

# --- 5. Generate ---
print("Generating...")
with torch.no_grad():
    output = model.generate(
        encodings.input_ids,
        attention_mask=encodings.attention_mask,
        max_new_tokens=100,      # We only need a name and 1 sentence
        do_sample=True,
        temperature=0.2,         # Low temp = Logical, Focused
        top_p=0.90,              # Standard filtering
        repetition_penalty=1.15, # Prevents "Step 2 Name: Step 2 Name:" loops
        pad_token_id=tokenizer.eos_token_id
    )

# --- 6. Output ---
# Decode only the new tokens
new_tokens = output[0][encodings.input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("="*20 + " Result " + "="*20)
print(f"{response.strip()}") 
print("="*48)

# --- 7. Cleanup ---
del model, tokenizer, encodings, output
gc.collect()
torch.cuda.empty_cache()