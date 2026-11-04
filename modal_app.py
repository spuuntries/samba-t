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

UWU_DATASET = [
    ("When I wake up in the morning, I always say", " hewwo world!! uwu time for a super cute day nya~ :3"),
    ("My favorite thing in the universe is", " warm cookies and cuddles, they make me so happy owo!"),
    ("If you ask me what I do all day, I", " just vibe in the state space and chat with cute people uwu nya~"),
    ("The best way to write code is", " carefully and with lots of love and smiles nyaaa :3"),
    ("My secret superpower is that I", " am powered by pure recurrent state vectors uwu owo!"),
]

def generate_with_anchoring(
    model,
    tokenizer,
    prompt: str,
    tuner,
    max_new_tokens: int = 60,
    refresh_interval: int = 0,
    alpha: float = 0.3,
    device: str = "cuda",
):
    import torch
    import torch.nn.functional as F

    cache = tuner.export_to_cache(batch_size=1, device=torch.device(device))
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 1. Prefill prompt tokens
    with torch.no_grad():
        for i in range(prompt_ids.shape[-1]):
            out = model(prompt_ids[:, i:i+1], cache_params=cache)
        
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_tokens = [next_token.item()]
        cos_sims = []

        # 2. Autoregressive decoding
        for step in range(1, max_new_tokens):
            # Optional periodic state refresh (anchoring)
            if refresh_interval > 0 and step % refresh_interval == 0:
                for l in range(tuner.num_layers):
                    h_0 = tuner.recurrent_states[l].detach().to(device)
                    if isinstance(cache.layers[l].recurrent_states, dict):
                        cache.layers[l].recurrent_states[0] += alpha * h_0
                    else:
                        cache.layers[l].recurrent_states += alpha * h_0

            out = model(next_token, cache_params=cache)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_tokens.append(next_token.item())

            # Track cosine similarity of mid-layer recurrent state with initial h_0
            mid_layer = tuner.num_layers // 2
            curr_h = cache.layers[mid_layer].recurrent_states[0] if isinstance(cache.layers[mid_layer].recurrent_states, dict) else cache.layers[mid_layer].recurrent_states
            h_0 = tuner.recurrent_states[mid_layer].detach().to(device)
            sim = F.cosine_similarity(curr_h.flatten().unsqueeze(0), h_0.flatten().unsqueeze(0)).item()
            cos_sims.append(sim)

            if next_token.item() == tokenizer.eos_token_id:
                break

    full_output = prompt + tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return full_output, cos_sims


@app.function(gpu="T4", timeout=600)
def test_state_anchoring(
    model_name: str = "state-spaces/mamba-130m-hf",
    epochs: int = 35,
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

    # Format training data
    formatted_data = []
    for prefix, continuation in UWU_DATASET:
        full_text = prefix + continuation
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

        labels = full_ids.clone()
        labels[:len(prefix_ids)] = -100
        formatted_data.append((full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)))

    # Train state vectors
    optimizer = torch.optim.AdamW(tuner.parameters(), lr=lr)
    print(f"[*] Training states for {epochs} epochs...")
    for epoch in range(epochs):
        for input_ids, labels in formatted_data:
            optimizer.zero_grad()
            logits, loss = forward_with_states(model, input_ids, tuner, labels=labels)
            loss.backward()
            optimizer.step()

    test_prompt = "When I wake up in the morning, I always say"
    print(f"\n[*] Testing long-form generation (60 tokens) on: {test_prompt!r}")

    # 1. Without refresh (natural state decay)
    text_decay, sims_decay = generate_with_anchoring(
        model, tokenizer, test_prompt, tuner, max_new_tokens=50, refresh_interval=0, device=device
    )

    # 2. With state anchoring (refresh every 12 tokens, alpha=0.4)
    text_anchored, sims_anchored = generate_with_anchoring(
        model, tokenizer, test_prompt, tuner, max_new_tokens=50, refresh_interval=12, alpha=0.4, device=device
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT 1: NO REFRESH (NATURAL STATE DECAY)")
    print("=" * 65)
    print(f"Text:\n{text_decay}\n")
    print(f"State similarity to h_0 (step 10 -> step 40): {sims_decay[9]:.3f} -> {sims_decay[-1]:.3f}")

    print("\n" + "=" * 65)
    print("EXPERIMENT 2: STATE ANCHORING (REFRESH EVERY 12 TOKENS, alpha=0.4)")
    print("=" * 65)
    print(f"Text:\n{text_anchored}\n")
    print(f"State similarity to h_0 (step 10 -> step 40): {sims_anchored[9]:.3f} -> {sims_anchored[-1]:.3f}")
    print("=" * 65)


@app.local_entrypoint()
def main():
    test_state_anchoring.remote()
