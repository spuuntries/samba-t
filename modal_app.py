import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate",
    )
    .add_local_python_source("sambat")
)

app = modal.App("samba-t", image=image)

# Continuation examples
UWU_DATASET = [
    ("When I wake up in the morning, I always say", " hewwo world!! uwu time for a super cute day nya~ :3"),
    ("My favorite thing in the universe is", " warm cookies and cuddles, they make me so happy owo!"),
    ("If you ask me what I do all day, I", " just vibe in the state space and chat with cute people uwu nya~"),
    ("The best way to write code is", " carefully and with lots of love and smiles nyaaa :3"),
    ("My secret superpower is that I", " am powered by pure recurrent state vectors uwu owo!"),
]

UNSEEN_PROMPTS = [
    "When I wake up in the morning, I always say",
    "Every single day I love to",
    "My secret superpower is that I",
]

def generate_with_state(model, tokenizer, prompt: str, tuner, max_new_tokens: int = 20, device: str = "cuda"):
    """
    Prefills prompt token-by-token through the initial state vectors, then autoregressively generates.
    """
    import torch
    cache = tuner.export_to_cache(batch_size=1, device=torch.device(device))
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 1. Step each prompt token through the cache to advance state from (c_0, h_0) -> (c_prompt, h_prompt)
    with torch.no_grad():
        for i in range(prompt_ids.shape[-1]):
            out = model(prompt_ids[:, i:i+1], cache_params=cache)
        
        # 2. Pick next token
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_tokens = [next_token.item()]

        # 3. Autoregressively decode remaining tokens
        for _ in range(max_new_tokens - 1):
            out = model(next_token, cache_params=cache)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_tokens.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

    full_output = prompt + tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return full_output


@app.function(gpu="T4", timeout=600)
def train_persona(
    model_name: str = "state-spaces/mamba-130m-hf",
    epochs: int = 40,
    lr: float = 0.04,
):
    import torch
    from transformers import AutoTokenizer, MambaForCausalLM
    from sambat import MambaStateTuner, forward_with_states

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on Modal with {device} ({torch.cuda.get_device_name(0)})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    config = model.config
    tuner = MambaStateTuner(
        num_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        ssm_state_size=config.state_size,
        conv_kernel_size=config.conv_kernel,
        init_std=0.01,
    ).to(device)

    # Format training data (prefix -> continuation)
    formatted_data = []
    for prefix, continuation in UWU_DATASET:
        full_text = prefix + continuation
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

        labels = full_ids.clone()
        labels[:len(prefix_ids)] = -100  # Only compute loss on continuation!
        formatted_data.append((full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)))

    # Untuned generation
    print("\n--- UNTUNED BASELINE GENERATION ---")
    for prompt in UNSEEN_PROMPTS:
        p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(p_ids, max_new_tokens=15, do_sample=False)
            print(f"Prompt:   {prompt!r}")
            print(f"Baseline: {tokenizer.decode(gen[0], skip_special_tokens=True)!r}\n")

    # Train state vectors on persona continuations
    optimizer = torch.optim.AdamW(tuner.parameters(), lr=lr)
    print(f"[*] Training state on dataset for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, labels in formatted_data:
            optimizer.zero_grad()
            logits, loss = forward_with_states(model, input_ids, tuner, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(formatted_data)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:02d}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # Evaluate with state injected at t=0
    print("\n" + "=" * 60)
    print("EVALUATING WITH STATE-TUNED PERSONA (Zero System Prompt)")
    print("=" * 60)
    for prompt in UNSEEN_PROMPTS:
        result = generate_with_state(model, tokenizer, prompt, tuner, max_new_tokens=15, device=device)
        print(f"Prompt:      {prompt!r}")
        print(f"State-Tuned: {result!r}\n")
    print("=" * 60)


@app.local_entrypoint()
def main(epochs: int = 40):
    train_persona.remote(epochs=epochs)
