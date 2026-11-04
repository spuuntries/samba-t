# Sambat: Stateful Mamba Tuning 🐍✨

Zero-overhead, zero-token prompt conditioning for Mamba State-Space Models (SSMs).

Inspired by RWKV state tuning, **Sambat** completely freezes the underlying Mamba model and optimizes *only* the initial recurrent state vectors ($h_0$) and convolutional history buffers ($c_0$) across all layers.

---

### Why State Tuning for Mamba?

- **0 Context Window Overhead:** Unlike soft prompts or prefix tokens, state tuning does not consume context window tokens.
- **0 Inference Compute Penalty:** Once tuned, initial states are directly injected into Hugging Face's `Cache` at $t=0$. Inference runs at native speed with zero extra FLOPs.
- **Microscopic Footprint:** The learned state vectors represent $< 0.6\%$ of the base model parameter size and can be saved/loaded as a tiny `.pt` checkpoint.

---

### How it Works

Mamba processes sequential inputs across each layer using:
1. **1D Causal Convolution:** A sliding window with kernel size $K=4$, buffering $K-1=3$ historical hidden representations.
2. **Selective SSM:** A continuous-to-discrete recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$.

During standard forward passes, $h_0 = 0$ and conv states are zero-padded. Samba-T provides:
- A fully differentiable sequence-level forward pass (`forward_with_states`) using exact causal history padding for $c_0$ and SSM recurrence initialization for $h_0$.
- Export to Hugging Face `Cache` (`export_to_cache`) with mathematical float-precision equivalence ($< 10^{-7}$ diff) to native decoding.

---

### Quickstart

#### Run on Modal (Cloud GPU):
```bash
modal run modal_app.py --steps 40
```

#### Run Locally:
```bash
python demo.py --model state-spaces/mamba-130m-hf --target "Hello! I am a state-tuned Mamba model running with zero prompt tokens."
```

#### Run Unit Tests:
```bash
python tests/test_e2e.py
```

---

### Library Usage

```python
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from sambat import MambaStateTuner, forward_with_states

# 1. Load model & freeze weights
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").eval()
for p in model.parameters():
    p.requires_grad = False

# 2. Instantiate tuner (< 0.6% params)
tuner = MambaStateTuner(
    num_layers=model.config.num_hidden_layers,
    intermediate_size=model.config.intermediate_size,
    ssm_state_size=model.config.state_size,
    conv_kernel_size=model.config.conv_kernel,
)

# 3. Train states via backprop
optimizer = torch.optim.AdamW(tuner.parameters(), lr=0.05)
for step in range(40):
    optimizer.zero_grad()
    logits, loss = forward_with_states(model, input_ids, tuner, labels=input_ids)
    loss.backward()
    optimizer.step()

# 4. Generate with zero extra prompt tokens!
cache = tuner.export_to_cache(batch_size=1)
out = model.generate(torch.tensor([[1]]), cache_params=cache, max_new_tokens=20)
```
