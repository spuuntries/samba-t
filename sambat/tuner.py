import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, LinearAttentionLayer


class MambaStateTuner(nn.Module):
    """
    Learns initial recurrent states (h_0) and conv states (c_0) for each layer of a Mamba model.
    All original model weights remain frozen; only these state vectors are optimized.
    """
    def __init__(
        self,
        num_layers: int,
        intermediate_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        init_std: float = 0.02,
        tune_conv: bool = True,
        tune_ssm: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.tune_conv = tune_conv
        self.tune_ssm = tune_ssm

        # Recurrent SSM state per layer: [intermediate_size, ssm_state_size]
        self.recurrent_states = nn.ParameterList([
            nn.Parameter(
                torch.randn(intermediate_size, ssm_state_size) * init_std if tune_ssm else torch.zeros(intermediate_size, ssm_state_size),
                requires_grad=tune_ssm
            )
            for _ in range(num_layers)
        ])

        # Conv state history per layer: [intermediate_size, conv_kernel_size - 1]
        self.conv_states = nn.ParameterList([
            nn.Parameter(
                torch.randn(intermediate_size, conv_kernel_size - 1) * init_std if tune_conv else torch.zeros(intermediate_size, conv_kernel_size - 1),
                requires_grad=tune_conv
            )
            for _ in range(num_layers)
        ])

    def export_to_cache(self, batch_size: int = 1, device: torch.device | None = None) -> Cache:
        """
        Exports the learned states into a Hugging Face `Cache` for zero-overhead inference with `model.generate`.
        Handles both single-tensor and multi-head dict cache schemas across transformers versions.
        """
        layers = []
        for l in range(self.num_layers):
            layer_cache = LinearAttentionLayer()
            
            # Prepare conv cache tensor of shape [batch, intermediate_size, conv_kernel_size]
            c = self.conv_states[l].detach()
            if device is not None:
                c = c.to(device)
            k = self.conv_kernel_size
            conv_tensor = torch.zeros((batch_size, self.intermediate_size, k), dtype=c.dtype, device=c.device)
            conv_tensor[:, :, -(k - 1):] = c.unsqueeze(0).expand(batch_size, -1, -1)

            # Prepare recurrent SSM state tensor of shape [batch, intermediate_size, ssm_state_size]
            h = self.recurrent_states[l].detach()
            if device is not None:
                h = h.to(device)
            h_tensor = h.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

            # Initialize shapes and static address via lazy_init
            layer_cache.lazy_initialization(conv_states=conv_tensor, recurrent_states=h_tensor)

            # Check if cache layer uses dict storage (newer HF) or direct attribute (older HF)
            if isinstance(layer_cache.conv_states, dict):
                layer_cache.conv_states[0].copy_(conv_tensor)
                layer_cache.recurrent_states[0].copy_(h_tensor)
                if isinstance(layer_cache.has_previous_state, dict):
                    layer_cache.has_previous_state[0] = True
                else:
                    layer_cache.has_previous_state = True
            else:
                layer_cache.conv_states.copy_(conv_tensor)
                layer_cache.recurrent_states.copy_(h_tensor)
                layer_cache.has_previous_state = True

            layers.append(layer_cache)

        return Cache(layers=layers)

    def save_state(self, filepath: str):
        torch.save({
            "recurrent_states": [p.detach().cpu() for p in self.recurrent_states],
            "conv_states": [p.detach().cpu() for p in self.conv_states],
            "config": {
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "ssm_state_size": self.ssm_state_size,
                "conv_kernel_size": self.conv_kernel_size,
            }
        }, filepath)

    def load_state(self, filepath: str, device: torch.device | None = None):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        for p, loaded in zip(self.recurrent_states, checkpoint["recurrent_states"]):
            p.data.copy_(loaded)
            if device is not None:
                p.data = p.data.to(device)
        for p, loaded in zip(self.conv_states, checkpoint["conv_states"]):
            p.data.copy_(loaded)
            if device is not None:
                p.data = p.data.to(device)


def mixer_forward_with_states(mixer, input_states, c_0, h_0):
    """
    Differentiable MambaMixer sequence forward pass supporting custom initial states.
    input_states: [batch, seq_len, hidden_size]
    c_0: [batch, intermediate_size, conv_kernel_size - 1]
    h_0: [batch, intermediate_size, ssm_state_size]
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # 1. Gated MLP projection
    projected_states = mixer.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    # 2. Convolution with initial conv state c_0 as history padding (padding=0 for exact match)
    conv_input = torch.cat([c_0, hidden_states], dim=-1)
    conv_out = F.conv1d(
        conv_input,
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        groups=mixer.intermediate_size,
        padding=0
    )
    hidden_states = mixer.act(conv_out)

    # 3. SSM sequence transformation
    ssm_parameters = mixer.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1
    )
    discrete_time_step = mixer.dt_proj(time_step)
    discrete_time_step = F.softplus(discrete_time_step).transpose(1, 2)

    A = -torch.exp(mixer.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # Recurrence with initial state h_0
    scan_outputs = []
    ssm_state = h_0
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)

    scan_output = scan_output + (hidden_states * mixer.D[None, :, None])
    scan_output = scan_output * mixer.act(gate)

    contextualized_states = mixer.out_proj(scan_output.transpose(1, 2))
    return contextualized_states


def forward_with_states(
    model,
    input_ids: torch.LongTensor,
    tuner: MambaStateTuner,
    labels: torch.LongTensor | None = None,
):
    """
    Differentiable full Mamba causal LM forward pass conditioned on tuner's states.
    Returns (logits, loss).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Embed tokens
    hidden_states = model.backbone.embeddings(input_ids)

    # Process each Mamba layer
    for l, layer in enumerate(model.backbone.layers):
        c_0 = tuner.conv_states[l].unsqueeze(0).expand(batch_size, -1, -1)
        h_0 = tuner.recurrent_states[l].unsqueeze(0).expand(batch_size, -1, -1)

        residual = hidden_states
        normed_states = layer.norm(hidden_states.to(dtype=layer.norm.weight.dtype))
        if layer.residual_in_fp32:
            residual = residual.to(torch.float32)

        mixed_states = mixer_forward_with_states(layer.mixer, normed_states, c_0, h_0)
        hidden_states = residual + mixed_states

    # Final norm and LM head
    hidden_states = model.backbone.norm_f(hidden_states)
    logits = model.lm_head(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))

    return logits, loss


def export_to_cache(tuner: MambaStateTuner, batch_size: int = 1, device: torch.device | None = None) -> Cache:
    return tuner.export_to_cache(batch_size=batch_size, device=device)
