"""Conservative E(3) scalar/polar-vector message passing, implemented in torch."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .chemistry import descriptor_table


@dataclass
class ModelConfig:
    scalar_channels: int = 64
    vector_channels: int = 32
    blocks: int = 4
    radial_basis: int = 32
    cutoff: float = 5.0
    activation: str = "swiglu"
    average_neighbors: float = 16.0
    chemical_descriptors: bool = False
    auxiliary_heads: bool = False
    attention: bool = False
    attention_heads: int = 4
    tensor_channels: int = 0


class ScalarMLP(nn.Module):
    def __init__(self, input_dim, output_dim, width, activation):
        super().__init__()
        self.activation = activation
        if activation not in ("swiglu", "geglu", "silu"):
            raise ValueError(f"Unknown activation: {activation}")
        # Match parameter budgets across gated and ordinary scalar MLPs.
        hidden = max(1, round(width * (input_dim + output_dim + 1) /
                              ((2 if activation != "silu" else 1) * (input_dim + 1) + output_dim)))
        self.first = nn.Linear(input_dim, hidden * (1 if activation == "silu" else 2))
        self.last = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.first(x)
        if self.activation == "silu":
            x = F.silu(x)
        else:
            gate, value = x.chunk(2, dim=-1)
            x = (F.silu(gate) if self.activation == "swiglu" else F.gelu(gate)) * value
        return self.last(x)


def envelope(x):
    # Quintic smoothstep complement: f, f', f'' vanish at x=1.
    # Factored form avoids cancellation/negative weights near 1 in float32.
    return torch.where(x < 1, (1-x)**3 * (1 + 3*x + 6*x**2), torch.zeros_like(x))


class Interaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        s, v = config.scalar_channels, config.vector_channels
        self.s, self.v = s, v
        self.t = config.tensor_channels
        t = self.t
        self.attention = config.attention
        if self.attention:
            self.heads = config.attention_heads
            if s % self.heads or v % self.heads or t % self.heads:
                raise ValueError("Scalar/vector channels must be divisible by attention heads")
            self.query = nn.Linear(s, s, bias=False)
            self.key = nn.Linear(s, s, bias=False)
            self.attention_radial = nn.Linear(config.radial_basis, self.heads)
        self.norm = nn.LayerNorm(s)
        self.source = ScalarMLP(s, s + 2*v + 2*t, 2*s, config.activation)
        self.target_gate = nn.Linear(s, s + 2*v + 2*t)
        self.radial = nn.Linear(config.radial_basis, s + 2*v + 2*t)
        self.aggregation_scale = math.sqrt(config.average_neighbors)
        self.vector_mix = nn.Linear(v, v, bias=False)
        self.scalar_update = ScalarMLP(s + v + t, s + v + t, 2*s, config.activation)
        self.vector_a = nn.Linear(v, v, bias=False)
        self.vector_b = nn.Linear(v, v, bias=False)
        if t:
            self.tensor_mix = nn.Linear(t,t,bias=False)
            self.tensor_a = nn.Linear(t,t,bias=False)
            self.tensor_b = nn.Linear(t,t,bias=False)

    def forward(self, scalar, vector, radial, direction, cutoff, i, j, tensor=None):
        normalized = self.norm(scalar)
        central = 2 * torch.sigmoid(self.target_gate(normalized))[i]
        coefficients = self.source(normalized)[j] * central * self.radial(radial) * cutoff[:, None]
        ms, mv, md, mt, mq = torch.split(coefficients, (self.s, self.v, self.v, self.t, self.t), dim=-1)
        if self.attention:
            dimension = self.s // self.heads
            query = self.query(normalized).reshape(-1, self.heads, dimension)
            key = self.key(normalized).reshape(-1, self.heads, dimension)
            logits = (query[i] * key[j]).sum(-1) / math.sqrt(dimension) + self.attention_radial(radial)
            # Bounded logits avoid overflow without a neighbor-dependent maximum.
            exponent = torch.exp(5 * torch.tanh(logits / 5))
            weight = cutoff[:, None] * exponent
            denominator = weight.new_ones((len(scalar), self.heads)).index_add(0, i, weight)
            # A fixed null/self anchor in the denominator ensures the final
            # neighbor also vanishes smoothly as its distance reaches cutoff.
            factor = self.aggregation_scale * exponent / denominator[i]
            ms = ms * factor.repeat_interleave(self.s // self.heads, dim=-1)
            mv = mv * factor.repeat_interleave(self.v // self.heads, dim=-1)
            md = md * factor.repeat_interleave(self.v // self.heads, dim=-1)
            if self.t:
                tfactor = factor.repeat_interleave(self.t // self.heads, dim=-1)
                mt, mq = mt * tfactor, mq * tfactor
        # Vectors have [atom, xyz, channel] shape; channel mixing is equivariant.
        messages = self.vector_mix(vector)[j] * mv[:, None, :] + direction[:, :, None] * md[:, None, :]
        scalar = scalar + torch.zeros_like(scalar).index_add(0, i, ms) / self.aggregation_scale
        vector = vector + torch.zeros_like(vector).index_add(0, i, messages) / self.aggregation_scale
        a, b = self.vector_a(vector), self.vector_b(vector)
        invariant = (a * b).sum(dim=1)
        invariants = [self.norm(scalar), invariant]
        if self.t:
            quadrupole = direction[:,:,None] * direction[:,None,:] - torch.eye(3,device=direction.device,dtype=direction.dtype)[None] / 3
            tm = self.tensor_mix(tensor)[j] * mt[:,None,None,:] + quadrupole[:,:,:,None] * mq[:,None,None,:]
            tensor = tensor + torch.zeros_like(tensor).index_add(0,i,tm) / self.aggregation_scale
            ta, tb = self.tensor_a(tensor), self.tensor_b(tensor)
            invariants.append((ta * tb).sum(dim=(1,2)))
        ds, gate, tgate = torch.split(self.scalar_update(torch.cat(invariants, dim=-1)),
                                     (self.s,self.v,self.t), dim=-1)
        if self.t:
            tensor = tensor + ta * tgate[:,None,None,:] / math.sqrt(2)
        return scalar + ds / math.sqrt(2), vector + a * gate[:, None, :] / math.sqrt(2), tensor


class Potential(nn.Module):
    def __init__(self, config=None, offsets=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.embedding = nn.Embedding(119, c.scalar_channels)
        if c.chemical_descriptors:
            self.register_buffer("chemical_table", descriptor_table())
            self.chemical_projection = nn.Linear(4, c.scalar_channels, bias=False)
        if c.auxiliary_heads:
            self.auxiliary_readout = ScalarMLP(c.scalar_channels, 2, c.scalar_channels, c.activation)
        self.interactions = nn.ModuleList([Interaction(c) for _ in range(c.blocks)])
        self.readout = ScalarMLP(c.scalar_channels, 1, c.scalar_channels, c.activation)
        self.register_buffer("centers", torch.linspace(0, c.cutoff, c.radial_basis))
        self.register_buffer("offsets", torch.zeros(119))
        if offsets:
            for z, value in offsets.items():
                self.offsets[int(z)] = value
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def energy(self, graph, positions=None, cell=None, return_auxiliary=False):
        p = graph["positions"] if positions is None else positions
        cell = graph["cell"] if cell is None else cell
        i, j = graph["i"], graph["j"]
        d = p[j] - p[i] + torch.einsum("ei,eij->ej", graph["shifts"], cell[graph["edge_batch"]])
        distance = torch.linalg.vector_norm(d, dim=-1)
        direction = d / distance[:, None].clamp_min(1e-12)
        width = self.config.cutoff / self.config.radial_basis
        radial = torch.exp(-0.5 * ((distance[:, None] - self.centers) / width)**2)
        cutoff = envelope(distance / self.config.cutoff)
        scalar = self.embedding(graph["z"])
        if self.config.chemical_descriptors:
            scalar = scalar + self.chemical_projection(self.chemical_table[graph["z"]])
        vector = p.new_zeros((len(p), 3, self.config.vector_channels))
        tensor = p.new_zeros((len(p),3,3,self.config.tensor_channels)) if self.config.tensor_channels else None
        for interaction in self.interactions:
            scalar, vector, tensor = interaction(scalar, vector, radial, direction, cutoff, i, j, tensor)
        atom_energy = self.readout(scalar).squeeze(-1) + self.offsets[graph["z"]]
        energy = p.new_zeros(len(cell)).index_add(0, graph["batch"], atom_energy)
        # Keep isolated structures differentiable with exactly zero derivatives.
        energy = energy + 0 * (p.sum() + cell.sum())
        if return_auxiliary:
            return energy, self.auxiliary_readout(scalar) if self.config.auxiliary_heads else None
        return energy

    def forward(self, graph, create_graph=False, compute_stress=True):
        with torch.enable_grad():
            p = graph["positions"].detach().requires_grad_(True)
            cell = graph["cell"]
            if compute_stress:
                strain = torch.zeros_like(cell, requires_grad=True)
                deformation = torch.eye(3, dtype=p.dtype, device=p.device) + (strain + strain.transpose(-1, -2)) / 2
                moved = torch.einsum("ni,nij->nj", p, deformation[graph["batch"]])
                deformed_cell = cell @ deformation
                energy, auxiliary = self.energy(graph, moved, deformed_cell, return_auxiliary=True)
                dp, ds = torch.autograd.grad(energy.sum(), (p, strain), create_graph=create_graph)
                stress = ds / torch.linalg.det(cell).abs()[:, None, None]
            else:
                energy, auxiliary = self.energy(graph, p, cell, return_auxiliary=True)
                dp, = torch.autograd.grad(energy.sum(), (p,), create_graph=create_graph)
                stress = None
        return {"energy": energy, "forces": -dp, "stress": stress, "auxiliary": auxiliary}

    def configuration(self):
        return asdict(self.config)
