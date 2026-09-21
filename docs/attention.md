# Multi-head equivariant attention: 2025+ research and implementation

The non-attention baseline has four gated message-passing blocks. The optional
`--attention` variant has four stacked blocks, with four heads operating in
parallel inside each block. Attention is local to the complete periodic radius
graph, not a global all-atom transformer. Model depth and attention head count
are distinct configuration choices.

## Recent primary sources

- [eSEN, 2025](https://arxiv.org/abs/2502.12147): conservative forces and smooth,
  bounded energy derivatives must be tested with actual NVE integration.
  Hard neighbor truncation and unsmooth computational pathways are problematic.
- [UMA, 2025](https://arxiv.org/abs/2506.23971): modern universal models combine
  equivariant modeling with substantial data and scaling work. Their universal
  accuracy cannot be attributed to attention alone or extrapolated to our small
  subset and laptop-sized model.
- [EquiformerV3, April 2026](https://arxiv.org/abs/2604.09130): studies attention
  with a smooth radius cutoff, improved feedforward layers, and SwiGLU-S2
  activations for expressive equivariant modeling. This is particularly relevant
  to the user's attention/SwiGLU request. Our scalar SwiGLU is not its SwiGLU-S2
  construction, and our l=0/l=1 network is not a reproduction of EquiformerV3.
- [FlashTP, 2025](https://proceedings.mlr.press/v267/lee25l.html): tensor-product
  computation/memory motivates measuring resource costs as well as accuracy.

Older foundational references remain in the historical research report, but
these 2025+ papers guide the new attention implementation. All implementation
code remains our own PyTorch code; none of the referenced packages is imported.

## Implemented attention

For each head h, query/key projections operate on invariant scalar node features.
The attention logit combines their scaled dot product with a learned radial bias:

```
logit_ijh = dot(Q_h(s_i), K_h(s_j)) / sqrt(head_dim) + radial_bias_h(r_ij)
b_ijh = 5 tanh(logit_ijh / 5)
w_ijh = envelope(r_ij / cutoff) exp(b_ijh)
alpha_ijh = w_ijh / (1 + sum_k w_ikh)
```

The denominator includes a fixed zero-valued anchor. This is a cutoff-weighted
softmax-like local attention with an explicit null channel, not ordinary softmax
over a changing neighbor list. The anchor prevents the last neighbor's envelope
from cancelling between numerator and denominator. It also defines isolated
atoms without division by zero. Bounded logits avoid overflow without using a
neighbor-dependent max operation. The envelope has vanishing first and second
derivatives at the cutoff.

Each head weights a partition of scalar/vector message channels. Scalars weight
polar-vector values without acting on their Cartesian components. All messages,
including vector paths, are smoothly attenuated. Residual updates preserve a
node's own information. There is no coordinate-dependent hard top-k selection,
attention dropout, or batch-wide normalization.

This particular bounded-logit/null-anchor formula is a project design choice
motivated by smoothness requirements, not a claimed verbatim equation from any
of the cited architectures. Head weights are a model diagnostic, not proof of
physical bond order or causal importance.

## Training and evaluation

Train energy and its autograd-derived forces/stresses jointly. Do not train a
separate unconstrained force head for the MD checkpoint. AdamW, warmup/cosine,
gradient clipping and the same grouped data split apply to both variants.
SwiGLU/GeGLU apply to scalar feedforward pathways only.

Compare gated and attention variants with matched depth/width, data and update
budgets; report parameter counts because Q/K projections add parameters. A
single-seed pilot is exploratory and cannot establish a universal ranking.
Auxiliary DFT supervision and fixed elemental descriptors are separate switches
so attention gains are not confused with additional labels or input features.

Tests cover improper rotations, conservative derivatives, query/key gradient
flow, batch independence, and neighbor cutoff crossings both with other
neighbors present and for the last neighbor. Short NVE runs at two timesteps
provide an additional numerical check. None of these alone establishes
high-temperature, interface, or reactive accuracy.

Example:

```powershell
.\.venv\Scripts\python -m semi_mlip train --run runs/attention --attention --activation swiglu --loss pseudo_huber --epochs 50
```

## Optional rank-2 angular channels

The tensor experiment forms a symmetric traceless directional feature
`Q(u) = u u^T - I/3`. Messages combine learned channel mixtures of the neighbor
tensor with scalar-gated Q(u); only feature channels are mixed. The resulting
tensor transforms equivariantly under any orthogonal coordinate change, including
reflections. Scalar updates use Frobenius contractions between two channel
projections of the tensor. Scalar gates update tensors without altering their
Cartesian transformation law. Attention heads partition tensor channels just as
they partition vector channels.

This adds l=2 information that can survive inversion-symmetric neighborhoods
where vector sums cancel. It is a compact Cartesian construction, not a complete
set of Clebsch–Gordan tensor products or a copy of a cited architecture. Its
accuracy benefit must be established by the recorded ablations; extra angular
channels alone cannot repair missing training environments.
