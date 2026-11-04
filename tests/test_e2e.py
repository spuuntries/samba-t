import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import MambaConfig, MambaForCausalLM
from sambat import MambaStateTuner, forward_with_states

# 1. Setup small model
config = MambaConfig(
    vocab_size=100,
    hidden_size=64,
    state_size=16,
    num_hidden_layers=2,
    expand=2,
    conv_kernel=4,
    use_cache=True,
)
model = MambaForCausalLM(config).eval()
for p in model.parameters():
    p.requires_grad = False

# 2. Setup Tuner
tuner = MambaStateTuner(
    num_layers=config.num_hidden_layers,
    intermediate_size=config.intermediate_size,
    ssm_state_size=config.state_size,
    conv_kernel_size=config.conv_kernel,
    init_std=0.01,
)

# 3. Target sequence we want the model to generate from just token [1]
target_ids = torch.tensor([[1, 42, 73, 19, 88, 5]], dtype=torch.long)
optimizer = torch.optim.AdamW(tuner.parameters(), lr=0.1)

print("Starting State Tuning optimization...")
initial_loss = None
for step in range(40):
    optimizer.zero_grad()
    logits, loss = forward_with_states(model, target_ids, tuner, labels=target_ids)
    loss.backward()
    optimizer.step()
    
    if step == 0:
        initial_loss = loss.item()
    if step % 10 == 0 or step == 39:
        print(f"  Step {step:02d} | Loss: {loss.item():.4f}")

final_loss = loss.item()
assert final_loss < initial_loss * 0.5, "Loss should drop significantly during state tuning!"
print(f"Loss dropped from {initial_loss:.4f} -> {final_loss:.4f}!")

# 4. Generate starting from ONLY [1] with the tuned cache
cache = tuner.export_to_cache(batch_size=1)
input_id = torch.tensor([[1]], dtype=torch.long)
generated = model.generate(input_id, cache_params=cache, max_new_tokens=5, do_sample=False)
gen_tokens = generated[0].tolist()

print("\n--- RESULTS ---")
print("Target tokens:   ", target_ids[0].tolist())
print("Generated tokens:", gen_tokens)

assert gen_tokens == target_ids[0].tolist(), "Generated tokens should match target!"
print("PERFECT MATCH! Tuned state reproduced the exact sequence with zero prompt tokens!")

# 5. Test saving and loading state
state_path = os.path.join(os.path.dirname(__file__), "test_state.pt")
tuner.save_state(state_path)
print("State successfully saved to test_state.pt!")

tuner_loaded = MambaStateTuner(
    num_layers=config.num_hidden_layers,
    intermediate_size=config.intermediate_size,
    ssm_state_size=config.state_size,
    conv_kernel_size=config.conv_kernel,
)
tuner_loaded.load_state(state_path)
cache_loaded = tuner_loaded.export_to_cache(batch_size=1)
gen_loaded = model.generate(input_id, cache_params=cache_loaded, max_new_tokens=5, do_sample=False)
assert gen_loaded[0].tolist() == gen_tokens, "Loaded state should reproduce identical generation!"
print("State save/load verified identical!")

if os.path.exists(state_path):
    os.remove(state_path)

print("ALL TESTS PASSED!")
