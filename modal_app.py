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


@app.function(gpu="A10G", timeout=600)
def test_bigger_model(
    model_name: str = "state-spaces/mamba-1.4b-hf",
    epochs: int = 30,
    lr: float = 0.03,
):
    import torch
    from transformers import AutoTokenizer, MambaForCausalLM
    from sambat import MambaStateTuner, forward_with_states, generate_stateful

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on Modal with {device} ({torch.cuda.get_device_name(0)})...")

    print(f"[*] Loading {model_name} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    config = model.config
    print(f"[*] Base model frozen. Hidden size: {config.hidden_size}, Layers: {config.num_hidden_layers}")

    tuner = MambaStateTuner(
        num_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        ssm_state_size=config.state_size,
        conv_kernel_size=config.conv_kernel,
        init_std=0.01,
        dtype=torch.bfloat16,
    ).to(device)

    total_params = sum(p.numel() for p in tuner.parameters() if p.requires_grad)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Trainable state parameters: {total_params:,} ({total_params / base_params * 100:.3f}% of base model)")

    # Prepare training dataset
    formatted_data = []
    for prefix, continuation in UWU_DATASET:
        full_text = prefix + continuation
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

        labels = full_ids.clone()
        labels[:len(prefix_ids)] = -100
        formatted_data.append((full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)))

    test_prompt = "When I wake up in the morning, I always say"

    # Baseline untuned
    print(f"\n[*] Generating baseline from: {test_prompt!r}")
    untuned_text = generate_stateful(model, tokenizer, test_prompt, tuner=None, max_new_tokens=45, repetition_penalty=1.2, device=device)

    # Train state vectors on persona continuations
    optimizer = torch.optim.AdamW(tuner.parameters(), lr=lr)
    print(f"[*] Training state on 1.4B model for {epochs} epochs...")
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

    # Test 1: No refresh (natural decay)
    decay_text = generate_stateful(
        model, tokenizer, test_prompt, tuner=tuner, max_new_tokens=45, anchoring_mode="none", repetition_penalty=1.2, device=device
    )

    # Test 2: Smart boundary-aware refresh (only on punctuation like '.', '!', '~')
    boundary_text = generate_stateful(
        model, tokenizer, test_prompt, tuner=tuner, max_new_tokens=45, anchoring_mode="boundary", alpha=0.25, repetition_penalty=1.2, device=device
    )

    # Test 3: Unseen prompt
    unseen_prompt = "Every single day I love to"
    unseen_text = generate_stateful(
        model, tokenizer, unseen_prompt, tuner=tuner, max_new_tokens=45, anchoring_mode="boundary", alpha=0.25, repetition_penalty=1.2, device=device
    )

    print("\n" + "=" * 65)
    print("RESULTS ON MAMBA-1.4B:")
    print("=" * 65)
    print(f"1. UNTUNED BASELINE:\n{untuned_text}\n")
    print(f"2. STATE-TUNED (NO REFRESH - NATURAL DECAY):\n{decay_text}\n")
    print(f"3. STATE-TUNED (SMART BOUNDARY-AWARE ANCHORING):\n{boundary_text}\n")
    print(f"4. UNSEEN PROMPT (SMART BOUNDARY ANCHORING):\n{unseen_text}\n")
    print("=" * 65)


@app.local_entrypoint()
def main(model_name: str = "state-spaces/mamba-130m-hf"):
    test_bigger_model.remote(model_name=model_name)
