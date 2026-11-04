import argparse
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import AutoTokenizer, MambaForCausalLM
from sambat import MambaStateTuner, forward_with_states


def run_demo(
    model_name: str = "state-spaces/mamba-130m-hf",
    target_text: str = "Hello! I am a state-tuned Mamba model running with zero prompt tokens.",
    steps: int = 30,
    lr: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"[*] Loading tokenizer & model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()

    # Freeze base model weights
    for p in model.parameters():
        p.requires_grad = False

    config = model.config
    print(f"[*] Base model frozen. Hidden size: {config.hidden_size}, Layers: {config.num_hidden_layers}")

    # Initialize State Tuner
    tuner = MambaStateTuner(
        num_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        ssm_state_size=config.state_size,
        conv_kernel_size=config.conv_kernel,
        init_std=0.01,
    ).to(device)

    total_params = sum(p.numel() for p in tuner.parameters() if p.requires_grad)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Trainable state parameters: {total_params:,} ({total_params / base_params * 100:.3f}% of base model)")

    # Prepare training text
    input_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    print(f"[*] Target text ({input_ids.shape[-1]} tokens): {target_text!r}")

    # Show baseline generation (before tuning)
    start_id = input_ids[:, :1]
    with torch.no_grad():
        baseline_gen = model.generate(start_id, max_new_tokens=input_ids.shape[-1] - 1, do_sample=False)
        baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
    print(f"[Untuned Generation]: {baseline_text!r}")

    # Optimize initial state vectors
    optimizer = torch.optim.AdamW(tuner.parameters(), lr=lr)
    print(f"\n[*] Optimizing initial states for {steps} steps...")
    for step in range(steps):
        optimizer.zero_grad()
        logits, loss = forward_with_states(model, input_ids, tuner, labels=input_ids)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            print(f"    Step {step:02d}/{steps} | Loss: {loss.item():.4f}")

    # Export tuned state into HF Cache for inference
    cache = tuner.export_to_cache(batch_size=1, device=torch.device(device))

    # Generate starting from ONLY the first token
    with torch.no_grad():
        tuned_gen = model.generate(start_id, cache_params=cache, max_new_tokens=input_ids.shape[-1] - 1, do_sample=False)
        tuned_text = tokenizer.decode(tuned_gen[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print(f"Target Text:             {target_text}")
    print(f"Untuned Gen:             {baseline_text}")
    print(f"Tuned Gen (Zero Prompt): {tuned_text}")
    print("=" * 50)

    # Save state
    tuner.save_state("tuned_state.pt")
    print("[*] State saved to 'tuned_state.pt'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Samba-T: Mamba State Tuning Demo")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--target", type=str, default="Hello! I am a state-tuned Mamba model running with zero prompt tokens.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    run_demo(model_name=args.model, target_text=args.target, steps=args.steps, lr=args.lr)
