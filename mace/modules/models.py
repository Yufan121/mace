###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3, nn
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum, scatter_mean

import os

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

# pylint: disable=C0302


@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        # use_layer_norm: Optional[bool] = False,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            # use_layer_norm=use_layer_norm,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                # use_layer_norm=use_layer_norm,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                    )
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        save_intermediate: bool = False,
        save_dir: str = "./tsne_features",
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        num_graphs = data["ptr"].numel() - 1
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(   # scatter_sum: sum over the nodes in the graph
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        for i, (interaction, product, readout) in enumerate(zip(self.interactions, self.products, self.readouts)):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)
            if save_intermediate:
                # Save each layer's node features as a numpy file
                np.save(f"{save_dir}/layer_{i}_node_feats.npy", node_feats.detach().cpu().numpy())

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress, hessian = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        save_intermediate: bool = False,
        save_dir: str = "./tsne_features",
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)
        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for i, (interaction, product, readout) in enumerate(zip(self.interactions, self.products, self.readouts)):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats, node_heads)[num_atoms_arange, node_heads])

            if save_intermediate:
                # Save each layer's node features as a numpy file
                np.save(f"{save_dir}/layer_{i}_node_feats.npy", node_feats.detach().cpu().numpy())

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress, hessian = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )
        output = {
            "energy": total_e,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

        return output



import torch.nn.functional as F
class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.ln = torch.nn.LayerNorm(output_dim)
        # self.activation = torch.nn.Tanh()  # You can replace this with F.relu or other activation functions
        # self.activation = F.relu  # LeakyReLU  
        # self.activation = torch.nn.LeakyReLU(negative_slope=0.01)        
        self.activation = torch.nn.SiLU()
        
        # Add a linear layer to match dimensions if needed
        if input_dim != output_dim:
            self.match_dim = torch.nn.Linear(input_dim, output_dim)
        else:
            self.match_dim = None

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.ln(out)
        out = self.activation(out)
        
        if self.match_dim is not None:
            residual = self.match_dim(residual)
        
        out = out + residual # Use out-of-place addition        
        
        return out
    
    
# class ResidualBlockBS1(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ResidualBlockBS1, self).__init__()    
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.ln = nn.LayerNorm(output_dim)
#         # self.activation = nn.Tanh()  # You can replace this with F.relu or other activation functions
#         # self.activation = F.relu   
#         # self.activation = nn.LeakyReLU(negative_slope=0.01)        
#         self.activation = nn.SiLU()
        
#         # Add a linear layer to match dimensions if needed
#         if input_dim != output_dim:
#             self.match_dim = nn.Linear(input_dim, output_dim)
#         else:
#             self.match_dim = None

#     def forward(self, x):
#         residual = x
#         out = self.fc(x)
        
#         out = self.ln(out)
        
#         out = self.activation(out)
        
#         if self.match_dim is not None:
#             residual = self.match_dim(residual)
        
#         out = out + residual  # Use out-of-place addition        
        
#         return out

class PlusMinusSqrtIdentity(torch.nn.Module):
    def forward(self, x):
        # Apply identity function for the range -1 to 1
        identity_mask = (x >= -1) & (x <= 1)
        identity_output = x * identity_mask.float()
        
        # Apply ±sqrt function for other areas
        sqrt_mask = ~identity_mask
        sqrt_output = torch.sign(x) * torch.sqrt(torch.abs(x)) * sqrt_mask.float()
        
        # Combine the outputs
        output = identity_output + sqrt_output
        return output


# range constraint parameters
ele_param_enum = ['lev0', 'lev1', 'lev2', 'exp0', 'exp1', 'exp2', 'EN', 'GAM', 'GAM3', 'KCNS', 'KCNP', 'KCND', 'DPOL', 'QPOL', 'REPA', 'REPB', 'POLYS', 'POLYP', 'POLYD', 'LPARP', 'LPARD', 'mpvcn', 'mprad']

elem_param_included = ['REPB', 'GAM', 'GAM3', 'DPOL', 'QPOL', 'POLYS', 'POLYP']# , 'KCNS', 'KCNP']

elem_param_delta = {        #REPB,      GAM,      GAM3,     DPOL,       QPOL,       POLYS,      POLYP,    KCNS,    KCNP
                    'H':    [0.276347,	0.101443, 0.2,	    1.390972,	0.006858,	0.238405,	0.6], # last is unsure
                    'C':    [1.057769,	0.134504, 0.375,	0.102919,	0.053396,	0.57358, 	0.067776],
                    # 'N':    [1.310648,     0,      0,      0,      0,      0,      0,     0], 
                    'O':    [1.446104,	0.112974, 0.129283,	1.233918,	0.077707,	3.738823,	0.837705],
                    # 'F':    [1.755371,     0,      0,      0,      0,      0,      0,     0], 
                    # 'P':    [4.920876,     0,      0,      0,      0,      0,      0,     0], 
                    # 'S':    [3.748773,     0,      0,      0,      0,      0,      0,     0], 
                    # 'Cl':   [4.338283,   10,  10,  10,  10,  10,  10,  10,  10], 
                    # 'Br':   [8.211340,     0,      0,      0,      0,      0,      0,     0], 
                    # 'I':    [15.829794,     0,      0,      0,      0,      0,      0,     0], 
                    } 

@compile_mode("script")
class ScaleShiftMACExTB(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        outdim: int,
        outdim_globpar: int,
        outdim_pair: int,
        half_range_pt = None,
        half_range_pt_globpar = None,
        half_range_pt_pair = None,
        parallel_units_elempar: int = 128,
        parallel_units_globpar: int = 128,
        parallel_units_pair: int = 128,
        separate_output_heads: bool = True,
        use_custom_ranges: bool = False,
        use_scale_predictor: bool = False,
        scatter_method: str = "scatter_sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Initialize scale and shift block
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        
        # Set output dimensions
        self.outdim = outdim
        self.outdim_globpar = outdim_globpar
        self.outdim_pair = outdim_pair
        
        
        elemnet = False
        globnet = False
        pairnet = False
        if self.outdim > 0:
            elemnet = True
        if self.outdim_globpar > 0:
            globnet = True
        if self.outdim_pair > 0:
            pairnet = True
        self.elemnet = elemnet
        self.globnet = globnet
        self.pairnet = pairnet  
        
        # if outdim != ele_param_enum
        if use_custom_ranges and self.elemnet and self.outdim != len(ele_param_enum):
            raise ValueError(f"outdim {outdim} must be equal to the number of elements in ele_param_enum: {len(ele_param_enum)}")

        
        # only one of the three can be True
        # assert sum([elemnet, globnet, pairnet]) == 1, f"Only one of the three can be True: {elemnet}, {globnet}, {pairnet}"
        
        # Set parallel units
        self.parallel_units_elempar = parallel_units_elempar
        self.parallel_units_globpar = parallel_units_globpar
        self.parallel_units_pair = parallel_units_pair
        self.separate_output_heads = separate_output_heads
        self.use_custom_ranges = use_custom_ranges
        self.use_scale_predictor = use_scale_predictor
        self.scatter_method = scatter_method
        
        # Check if custom ranges can be used
        if self.use_custom_ranges and not self.separate_output_heads:
            raise ValueError("Elem & parameter custom ranges can only be used with separate_output_heads=True")
        
        # Initialize half range parameters
        self.half_range_pt = torch.tensor(half_range_pt or [0.05] * self.outdim, dtype=torch.get_default_dtype())
        self.half_range_pt_globpar = torch.tensor(half_range_pt_globpar or [0.05] * self.outdim_globpar, dtype=torch.get_default_dtype())
        self.half_range_pt_pair = torch.tensor(half_range_pt_pair or [0.05] * self.outdim_pair, dtype=torch.get_default_dtype())

        # Define connecting layer size
        # cnt_size = 256 + 35        # TODO
        cnt_size = 64 + 100
        
        # Create output heads
        if self.separate_output_heads:
            self.output_heads = self._create_output_heads(self.outdim, self.parallel_units_elempar, cnt_size)
            self.output_globpar_heads = self._create_output_heads(self.outdim_globpar, self.parallel_units_globpar, cnt_size)
            self.output_pair_heads = self._create_output_heads(self.outdim_pair, self.parallel_units_pair, cnt_size)
        else:
            self.output_head = self._create_combined_head(self.outdim, self.parallel_units_elempar, cnt_size)
            self.output_globpar_head = self._create_combined_head(self.outdim_globpar, self.parallel_units_globpar, cnt_size)
            self.output_pair_head = self._create_combined_head(self.outdim_pair, self.parallel_units_pair, cnt_size)
            
        # Define output activation
        self.out = torch.nn.Identity()
        
        # Create scale and shift predictors only if enabled
        if self.use_scale_predictor:
            self.scale_predictor = self._create_predictor(cnt_size, self.outdim)
            self.shift_predictor = self._create_predictor(cnt_size, self.outdim, use_softplus=False)
            self.scale_predictor_globpar = self._create_predictor(cnt_size, self.outdim_globpar)
            self.shift_predictor_globpar = self._create_predictor(cnt_size, self.outdim_globpar, use_softplus=False)
        else:
            self.scale_predictor = None
            self.shift_predictor = None
            self.scale_predictor_globpar = None
            self.shift_predictor_globpar = None

    def _create_output_heads(self, outdim, parallel_units, cnt_size, block_type=ResidualBlock):
        # Helper function to create separate output heads
        return torch.nn.ModuleList([
            torch.nn.Sequential(
                block_type(cnt_size, parallel_units),
                block_type(parallel_units, parallel_units),
                block_type(parallel_units, parallel_units),
                torch.nn.Linear(parallel_units, 1),
                # torch.nn.Tanh() if self.use_custom_ranges else torch.nn.Identity()
            ) for _ in range(outdim)
        ])

    def _create_combined_head(self, outdim, parallel_units, cnt_size):
        # Helper function to create a combined output head
        return torch.nn.Sequential(
            ResidualBlock(cnt_size, parallel_units),
            ResidualBlock(parallel_units, parallel_units),
            ResidualBlock(parallel_units, parallel_units),
            torch.nn.Linear(parallel_units, outdim)
        )

    def _create_predictor(self, cnt_size, outdim, use_softplus=True):
        # Helper function to create a predictor
        layers = [ 
            torch.nn.Linear(cnt_size, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 128),
            torch.nn.Linear(128, outdim)
        ]
        if use_softplus:
            layers.append(torch.nn.Softplus())
        return torch.nn.Sequential(*layers)

    def _print_grad_hook(self, grad: torch.Tensor, name: str):
        # Helper function for gradient hooks
        # print(f"--- Gradient Stats for {name} ---" + (f"\n  - Norm: {grad.norm().item():.4f}\n  - Mean: {grad.mean().item():.4f}\n  - Max Abs: {grad.abs().max().item():.4f}\n  - Has NaN: {torch.isnan(grad).any().item()}\n  - Has Inf: {torch.isinf(grad).any().item()}" if grad is not None else "\n--- No gradient computed for {name} ---"))
        print(f"--- Gradient Stats ---\n  - Norm: {grad.norm().item():.4f}\n  - Mean: {grad.mean().item():.4f}\n  - Max Abs: {grad.abs().max().item():.4f}\n  - Has NaN: {torch.isnan(grad).any().item()}\n  - Has Inf: {torch.isinf(grad).any().item()}")

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        output_indices = None,
        save_intermediate: bool = False,
        save_dir: str = "./tsne_features",
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup and gradient requirements
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        # print(f'node_attrs.shape: {data["node_attrs"].shape}, node_attrs: {data["node_attrs"]}')
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        node_heads = data.get("head", torch.zeros_like(data["batch"]))[data["batch"]]
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            data["positions"], data["shifts"], displacement = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Compute atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[num_atoms_arange, node_heads]
        e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs)

        # Compute embeddings and edge features
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)
        pair_node_energy = self.pair_repulsion_fn(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers) if hasattr(self, "pair_repulsion") else torch.zeros_like(node_e0)

        # Process interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for i, (interaction, product, readout) in enumerate(zip(self.interactions, self.products, self.readouts)):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats, node_heads)[num_atoms_arange, node_heads])

            # if training:
                # node_feats_list[-1].register_hook(lambda grad: self._print_grad_hook(grad, f"node_feats_list[-1]"))
                # node_es_list[-1].register_hook(lambda grad: self._print_grad_hook(grad, f"node_es_list[-1]"))
                
            

            if save_intermediate:
                # Save each layer's node features as a numpy file
                # make sure the directory exists
                os.makedirs(save_dir, exist_ok=True)
                if self.elemnet:
                    np.save(f"{save_dir}/layer_{i}_elemnet_node_feats.npy", node_feats.detach().cpu().numpy())
                elif self.globnet:
                    np.save(f"{save_dir}/layer_{i}_globnet_node_feats.npy", node_feats.detach().cpu().numpy())
                elif self.pairnet:
                    np.save(f"{save_dir}/layer_{i}_pairnet_node_feats.npy", node_feats.detach().cpu().numpy())

        # concatenate node features
        # print(f'len(node_feats_list): {len(node_feats_list)}')
        # for i in range(len(node_feats_list)):
        #     print(f'node_feats_list[{i}].shape: {node_feats_list[i].shape}')
        # # Concatenate node features
        # node_feats_out = torch.cat(node_feats_list, dim=-1)
        # print(f'node_feats_out.shape: {node_feats_out.shape}')
        # if training and node_feats_out.requires_grad:
        #     node_feats_out.register_hook(lambda grad: self._print_grad_hook(grad, "node_feats_out"))
        # only use the last node features
        node_feats_out = node_feats_list[-1]
        # print(f'node_feats_out.shape: {node_feats_out.shape}')
        
        # concat with node_attrs # TODO
        node_feats_out = torch.cat([node_feats_out, data["node_attrs"]], dim=-1)
        
        # print(f'node_feats_out.shape after concat: {node_feats_out.shape}')
        
        # Predict scale and shift only if enabled
        if self.use_scale_predictor:
            scale = self.scale_predictor(node_feats_out)
            shift = self.shift_predictor(node_feats_out)
        else:
            scale = None
            shift = None

        #### Compute element parameters #### 
        if self.separate_output_heads:
            outputs = [head(node_feats_out) for head in self.output_heads]
            params_pred_raw = torch.cat(outputs, dim=1) if outputs else None
        else:
            params_pred_raw = self.output_head(node_feats_out) if self.outdim > 0 else None

        if training and params_pred_raw is not None and params_pred_raw.requires_grad:  # debug hook
            params_pred_raw.register_hook(lambda grad: self._print_grad_hook(grad, "params_pred_raw"))
            
        # save params_pred_raw
        if save_intermediate:
            np.save(f"{save_dir}/params_pred_raw.npy", params_pred_raw.detach().cpu().numpy())

        params_pred = params_pred_raw
        if params_pred_raw is not None:
            if self.use_custom_ranges:
                # Get atomic numbers for each node
                atomic_numbers = self.atomic_numbers[data["node_attrs"].argmax(dim=1)]
                
                # Create a tensor of deltas for each node
                deltas = torch.zeros_like(params_pred_raw)
                for i, atom in enumerate(atomic_numbers):
                    atom_num = atom.item()
                    atom_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}.get(atom_num)
                    if atom_symbol in elem_param_delta:
                        # Get the delta values for this atom
                        atom_deltas = torch.tensor(elem_param_delta[atom_symbol], device=params_pred_raw.device)
                        # Map each parameter to its correct index in ele_param_enum
                        for j, param in enumerate(elem_param_included):
                            if param in ele_param_enum:
                                param_idx = ele_param_enum.index(param)
                                if param_idx < params_pred_raw.shape[1]:
                                    deltas[i, param_idx] = atom_deltas[j]
                    else:  # handle unknown atoms
                        raise ValueError(f'TODO: atom_symbol: {atom_symbol} not in elem_param_delta')
                
                print(f'deltas: {deltas}')
                
                # Apply deltas to included parameters, use half_range_pt for others
                half_range_pt_device = self.half_range_pt.to(params_pred_raw.device)
                # Create a mask for included parameters
                included_mask = torch.zeros_like(params_pred_raw, dtype=torch.bool)
                for param in elem_param_included:
                    if param in ele_param_enum:
                        param_idx = ele_param_enum.index(param)
                        if param_idx < params_pred_raw.shape[1]:
                            included_mask[:, param_idx] = True
                
                # Apply deltas to included parameters, half_range_pt to others
                params_pred = torch.where(
                    included_mask,
                    torch.nn.functional.tanh(params_pred_raw * 0.1) * deltas, # tanh for range constraint, stretch to tune
                    params_pred_raw * half_range_pt_device
                )
            else:
                params_pred = self.out(params_pred) * self.half_range_pt.to(params_pred.device)

        if output_indices is not None and params_pred is not None:
            params_pred = params_pred[output_indices]

        #### Compute global parameters #### 
        if self.scatter_method == "scatter_mean":
            global_feats = scatter_mean(src=node_feats_out, index=data["batch"], dim=0, dim_size=num_graphs)
        else:  # scatter_sum
            global_feats = scatter_sum(src=node_feats_out, index=data["batch"], dim=0, dim_size=num_graphs)

        if self.use_scale_predictor:
            scale_globpar = self.scale_predictor_globpar(global_feats)
            shift_globpar = self.shift_predictor_globpar(global_feats)
        else:
            scale_globpar = None
            shift_globpar = None

        if self.separate_output_heads:
            outputs_globpar = [head(global_feats) for head in self.output_globpar_heads]
            outputs_globpar_raw = torch.cat(outputs_globpar, dim=1) if outputs_globpar else None
        else:
            outputs_globpar_raw = self.output_globpar_head(global_feats) if self.outdim_globpar > 0 else None

        if training and outputs_globpar_raw is not None and outputs_globpar_raw.requires_grad: # debug hook
            outputs_globpar_raw.register_hook(lambda grad: self._print_grad_hook(grad, "outputs_globpar_raw"))

        outputs_globpar = outputs_globpar_raw
        if outputs_globpar_raw is not None:
            outputs_globpar = self.out(outputs_globpar) * self.half_range_pt_globpar.to(outputs_globpar.device)

        #### Compute pair parameters #### 
        pair_params = []
        if self.outdim_pair > 0:
            for graph_idx in range(num_graphs):
                start_idx = data['ptr'][graph_idx].item()
                end_idx = data['ptr'][graph_idx + 1].item()
                num_atoms = end_idx - start_idx
                graph_node_feats = node_feats_out[start_idx:end_idx]
                pairwise_feats = graph_node_feats.unsqueeze(0).expand(num_atoms, -1, -1) + graph_node_feats.unsqueeze(1).expand(-1, num_atoms, -1)
                pairwise_feats = pairwise_feats.reshape(num_atoms * num_atoms, -1)

                if self.separate_output_heads:
                    graph_pair_params = [head(pairwise_feats) for head in self.output_pair_heads]
                    graph_pair_params = torch.cat(graph_pair_params, dim=1)
                else:
                    graph_pair_params = self.output_pair_head(pairwise_feats) if self.output_pair_head is not None else torch.zeros(num_atoms * num_atoms, self.outdim_pair, device=pairwise_feats.device)

                graph_pair_params = self.out(graph_pair_params) * self.half_range_pt_pair.to(graph_pair_params.device)
                
                pair_params.append(graph_pair_params)
            
            pair_params = torch.stack(pair_params, dim=0)
            if pair_params.ndim == 2:
                pair_params = pair_params.unsqueeze(1)


        # Prepare output dictionary
        output = {
            "params_pred": params_pred,
            "globpars_pred": outputs_globpar,
            "node_feats": node_feats_out,
            "pair_param": pair_params,
        }

        return output



class BOTNet(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        gate: Optional[Callable],
        avg_num_neighbors: float,
        atomic_numbers: List[int],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True
        num_atoms_arange = torch.arange(data.positions.shape[0])

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=data.num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=data.num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=data["shifts"],
            cell=data["cell"],
            training=training,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": data["shifts"],
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


class ScaleShiftBOTNet(BOTNet):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True
        num_atoms_arange = torch.arange(data.positions.shape[0])
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es, data["head"][data["batch"]])

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e, positions=data.positions, training=training
            ),
        }

        return output


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, readout in zip(
            self.interactions, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


# Equivariant version of ResidualBlock
@compile_mode("script")
class EquivariantResidualBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, gate: Optional[Callable] = None):
        super(EquivariantResidualBlock, self).__init__()
        
        # Use e3nn Linear instead of nn.Linear
        self.linear = o3.Linear(irreps_in, irreps_out, internal_weights=True, shared_weights=True)
        
        # Use e3nn normalization instead of LayerNorm
        # self.norm = None  # Remove normalization for simplicity, or use e3nn.nn.BatchNorm
        
        # Use e3nn activation for scalars only
        if gate is None:
            gate = torch.nn.functional.silu
            
        # Create activation for scalars only
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
        if len(irreps_scalars) > 0:
            self.activation = nn.Activation(irreps_in=irreps_scalars, acts=[gate])
        else:
            self.activation = None
        
        # Residual connection - handle dimension mismatch
        if irreps_in != irreps_out:
            self.match_dim = o3.Linear(irreps_in, irreps_out, internal_weights=True, shared_weights=True)
        else:
            self.match_dim = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Linear transformation
        out = self.linear(x)
        
        # Apply activation only to scalar features
        if self.activation is not None:
            # Extract scalar features
            irreps_scalars = self.activation.irreps_in
            scalar_dim = irreps_scalars.dim
            
            if scalar_dim > 0:
                scalars = out[..., :scalar_dim]
                vectors = out[..., scalar_dim:]
                
                # Apply activation to scalars
                scalars = self.activation(scalars)
                
                # Concatenate back
                out = torch.cat([scalars, vectors], dim=-1)
        
        # Handle residual connection
        if self.match_dim is not None:
            residual = self.match_dim(residual)
        
        # Add residual
        out = out + residual
        
        return out


@compile_mode("script")
class EquivariantScaleShiftMACExTB(MACE):
    # 类级别的类型注解来消除 TorchScript 警告
    output_heads: Optional[torch.nn.ModuleList]
    output_globpar_heads: Optional[torch.nn.ModuleList]
    output_pair_heads: Optional[torch.nn.ModuleList]
    output_head: Optional[torch.nn.Module]
    output_globpar_head: Optional[torch.nn.Module]
    output_pair_head: Optional[torch.nn.Module]
    scale_predictor: Optional[torch.nn.Module]
    shift_predictor: Optional[torch.nn.Module]
    scale_predictor_globpar: Optional[torch.nn.Module]
    shift_predictor_globpar: Optional[torch.nn.Module]
    global_projection: Optional[torch.nn.Module]
    scalar_projection: Optional[torch.nn.Module]
    final_node_irreps: Optional[o3.Irreps]
    final_global_irreps: Optional[o3.Irreps]
    
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        outdim: int,
        outdim_globpar: int,
        outdim_pair: int,
        half_range_pt = None,
        half_range_pt_globpar = None,
        half_range_pt_pair = None,
        parallel_units_elempar: int = 100,
        parallel_units_globpar: int = 100,
        parallel_units_pair: int = 100,
        separate_output_heads: bool = True,
        separate_element_heads: bool = True,
        use_custom_ranges: bool = False,
        use_scale_predictor: bool = False,
        equivariant_feat_len: int = 64,  # Add predefined feature length
        scatter_method: str = "scatter_sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Initialize scale and shift block
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        
        # Set output dimensions
        self.outdim = outdim
        self.outdim_globpar = outdim_globpar
        self.outdim_pair = outdim_pair
        
        # Network flags
        elemnet = self.outdim > 0
        globnet = self.outdim_globpar > 0
        pairnet = self.outdim_pair > 0
        
        self.elemnet = elemnet
        self.globnet = globnet
        self.pairnet = pairnet
        
        # # Only one of the three can be True
        # assert sum([elemnet, globnet, pairnet]) == 1, f"Only one of the three can be True: {elemnet}, {globnet}, {pairnet}"
        
        # Set parallel units and other parameters
        self.parallel_units_elempar = parallel_units_elempar
        self.parallel_units_globpar = parallel_units_globpar
        self.parallel_units_pair = parallel_units_pair
        self.separate_output_heads = separate_output_heads
        self.separate_element_heads = separate_element_heads
        self.use_custom_ranges = use_custom_ranges
        self.use_scale_predictor = use_scale_predictor
        self.scatter_method = scatter_method
        
        # Check custom ranges compatibility
        if self.use_custom_ranges and not self.separate_output_heads:
            raise ValueError("Elem & parameter custom ranges can only be used with separate_output_heads=True")
        
        
        # Setup element-specific head information
        # Always set num_elements and element_to_idx first
        self.num_elements = len(self.atomic_numbers) if hasattr(self, 'atomic_numbers') else None
        self.element_to_idx = {z.item(): i for i, z in enumerate(self.atomic_numbers)} if hasattr(self, 'atomic_numbers') else None
        
        
        
        if self.num_elements is None or self.element_to_idx is None:
            raise ValueError("atomic_numbers is not set")

        
        # Initialize half range parameters and register as buffers for proper device handling
        if self.separate_element_heads and self.outdim > 0:
            # Create element-specific ranges: (num_elements, outdim)
            base_range = half_range_pt or [0.05] * self.outdim
            element_ranges = [base_range for _ in range(self.num_elements)]
            self.register_buffer(
                "half_range_pt",
                torch.tensor(element_ranges, dtype=torch.get_default_dtype())
            )
        else:
            # Original single range for all elements: (outdim,)
            self.register_buffer(
                "half_range_pt",
                torch.tensor(half_range_pt or [0.05] * self.outdim, dtype=torch.get_default_dtype())
            )
        
        self.register_buffer(
            "half_range_pt_globpar", 
            torch.tensor(half_range_pt_globpar or [0.05] * self.outdim_globpar, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "half_range_pt_pair",
            torch.tensor(half_range_pt_pair or [0.05] * self.outdim_pair, dtype=torch.get_default_dtype())
        )

        # Store the predefined feature length
        self.equivariant_feat_len = equivariant_feat_len
        
        # Set final irreps based on predefined feature length
        self.final_node_irreps = o3.Irreps(f"{equivariant_feat_len}x0e")  # Use predefined scalar features
        self.final_global_irreps = o3.Irreps("64x0e")  # Global scalar features
        
        
        if self.separate_element_heads and self.outdim > 0:
            # Reduce head size since we'll have num_elements * outdim heads
            # self.elempar_head_units = max(20, parallel_units_elempar // max(1, self.num_elements))
            self.elempar_head_units = 100
        else:
            self.elempar_head_units = parallel_units_elempar
        
        # Create scalar projection layer (from MACE features to our scalar features)
        # We'll assume MACE outputs have some scalar features, we'll project them
        self.scalar_projection = None  # Will be set if needed during forward
        
        # For global features, create projection layer
        if self.globnet:
            self.global_projection = o3.Linear(
                self.final_node_irreps,  # Input from node features
                self.final_global_irreps, 
                internal_weights=True, 
                shared_weights=True
            )
        else:
            self.global_projection = None
            
        # Create equivariant output heads immediately
        self._create_all_networks()
        
        # Define output activation
        self.out = torch.nn.Identity()

    def _create_all_networks(self):
        """Create all networks during initialization"""
        # Initialize all attributes to None first
        self.output_heads = None
        self.output_globpar_heads = None
        self.output_pair_heads = None
        self.output_head = None
        self.output_globpar_head = None
        self.output_pair_head = None
        
        # Create equivariant output heads
        if self.separate_output_heads:
            if self.outdim > 0:
                if self.separate_element_heads:
                    self.output_heads = self._create_element_specific_heads(
                        self.outdim, self.elempar_head_units, self.final_node_irreps
                    )
                else:
                    self.output_heads = self._create_equivariant_output_heads(
                        self.outdim, self.parallel_units_elempar, self.final_node_irreps
                    )
                
            if self.globnet:
                self.output_globpar_heads = self._create_equivariant_output_heads(
                    self.outdim_globpar, self.parallel_units_globpar, self.final_global_irreps
                )
                
            if self.pairnet:
                self.output_pair_heads = self._create_equivariant_output_heads(
                    self.outdim_pair, self.parallel_units_pair, self.final_node_irreps
                )
        else:
            if self.outdim > 0:
                if self.separate_element_heads:
                    self.output_head = self._create_element_specific_combined_heads(
                        self.outdim, self.elempar_head_units, self.final_node_irreps
                    )
                else:
                    self.output_head = self._create_equivariant_combined_head(
                        self.outdim, self.parallel_units_elempar, self.final_node_irreps
                    )
                
            if self.globnet:
                self.output_globpar_head = self._create_equivariant_combined_head(
                    self.outdim_globpar, self.parallel_units_globpar, self.final_global_irreps
                )
                
            if self.pairnet:
                self.output_pair_head = self._create_equivariant_combined_head(
                    self.outdim_pair, self.parallel_units_pair, self.final_node_irreps
                )
        
        # Create equivariant scale and shift predictors only if enabled
        if self.use_scale_predictor:
            if self.outdim > 0:
                self.scale_predictor = self._create_equivariant_predictor(self.final_node_irreps, self.outdim)
                self.shift_predictor = self._create_equivariant_predictor(self.final_node_irreps, self.outdim)
            else:
                self.scale_predictor = None
                self.shift_predictor = None
            
            if self.globnet:
                self.scale_predictor_globpar = self._create_equivariant_predictor(self.final_global_irreps, self.outdim_globpar)
                self.shift_predictor_globpar = self._create_equivariant_predictor(self.final_global_irreps, self.outdim_globpar)
            else:
                self.scale_predictor_globpar = None
                self.shift_predictor_globpar = None
        else:
            self.scale_predictor = None
            self.shift_predictor = None
            self.scale_predictor_globpar = None
            self.shift_predictor_globpar = None

    def _extract_and_project_scalars(self, node_feats):
        """Extract scalar features from MACE output and project to predefined dimension"""
        # MACE typically outputs features with both scalars and vectors
        # We need to extract just the scalar part and project it to our target dimension
        
        # Try to get the irreps from the last readout layer
        try:
            # Get the irreps from the last interaction's target_irreps
            last_interaction_irreps = self.interactions[-1].target_irreps
            scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in last_interaction_irreps if ir.l == 0])
            
            if scalar_irreps.dim > 0:
                # Extract scalar features
                scalar_features = node_feats[..., :scalar_irreps.dim]
                
                # If we need to project to a different dimension
                if scalar_irreps.dim != self.equivariant_feat_len:
                    if not hasattr(self, '_scalar_projection_layer') or self._scalar_projection_layer is None:
                        # Create projection layer on first use
                        self._scalar_projection_layer = torch.nn.Linear(
                            scalar_irreps.dim, 
                            self.equivariant_feat_len
                        ).to(node_feats.device)
                    scalar_features = self._scalar_projection_layer(scalar_features)
            else:
                raise ValueError("No scalar features found")
                # No scalar features, create them from a simple projection
                if not hasattr(self, '_scalar_creation_layer') or self._scalar_creation_layer is None:
                    self._scalar_creation_layer = torch.nn.Linear(
                        node_feats.shape[-1], 
                        self.equivariant_feat_len
                    ).to(node_feats.device)
                scalar_features = self._scalar_creation_layer(node_feats)
                
        except Exception:
            # Fallback: simple linear projection from full features
            if not hasattr(self, '_scalar_fallback_layer') or self._scalar_fallback_layer is None:
                self._scalar_fallback_layer = torch.nn.Linear(
                    node_feats.shape[-1], 
                    self.equivariant_feat_len
                ).to(node_feats.device)
            scalar_features = self._scalar_fallback_layer(node_feats)
            
        return scalar_features

    def _create_equivariant_output_heads(self, outdim, parallel_units, input_irreps):
        """Create separate equivariant output heads"""
        if outdim == 0:
            return torch.nn.ModuleList()
            
        # Create intermediate irreps for hidden layers (mainly scalars)
        hidden_irreps = o3.Irreps(f"{parallel_units}x0e")
        output_irreps = o3.Irreps("1x0e")  # Single scalar output
        
        heads = []
        for _ in range(outdim):
            head = torch.nn.Sequential(
                EquivariantResidualBlock(input_irreps, hidden_irreps),
                EquivariantResidualBlock(hidden_irreps, hidden_irreps),
                EquivariantResidualBlock(hidden_irreps, hidden_irreps),
                o3.Linear(hidden_irreps, output_irreps, internal_weights=True, shared_weights=True)
            )
            heads.append(head)
        
        return torch.nn.ModuleList(heads)

    def _create_equivariant_combined_head(self, outdim, parallel_units, input_irreps):
        """Create a combined equivariant output head"""
        if outdim == 0:
            return None
            
        # Create intermediate irreps
        hidden_irreps = o3.Irreps(f"{parallel_units}x0e")
        output_irreps = o3.Irreps(f"{outdim}x0e")  # Multiple scalar outputs
        
        return torch.nn.Sequential(
            EquivariantResidualBlock(input_irreps, hidden_irreps),
            EquivariantResidualBlock(hidden_irreps, hidden_irreps),
            EquivariantResidualBlock(hidden_irreps, hidden_irreps),
            o3.Linear(hidden_irreps, output_irreps, internal_weights=True, shared_weights=True)
        )

    def _create_equivariant_predictor(self, input_irreps, outdim):
        """Create an equivariant predictor"""
        if outdim == 0:
            return None
            
        # For scale/shift prediction, use smaller networks
        hidden_irreps = o3.Irreps("128x0e")
        output_irreps = o3.Irreps(f"{outdim}x0e")
        
        return torch.nn.Sequential(
            o3.Linear(input_irreps, hidden_irreps, internal_weights=True, shared_weights=True),
            nn.Activation(irreps_in=hidden_irreps, acts=[torch.nn.functional.silu]),
            o3.Linear(hidden_irreps, hidden_irreps, internal_weights=True, shared_weights=True),
            nn.Activation(irreps_in=hidden_irreps, acts=[torch.nn.functional.silu]),
            o3.Linear(hidden_irreps, output_irreps, internal_weights=True, shared_weights=True),
            nn.Activation(irreps_in=output_irreps, acts=[torch.nn.functional.softplus])  # For scale prediction
        )

    def _create_element_specific_heads(self, outdim, parallel_units, input_irreps):
        """Create element-specific separate output heads"""
        if outdim == 0:
            return torch.nn.ModuleList()
            
        # Create intermediate irreps for hidden layers (mainly scalars)
        hidden_irreps = o3.Irreps(f"{parallel_units}x0e")
        output_irreps = o3.Irreps("1x0e")  # Single scalar output
        
        heads = []
        # Create num_elements * outdim heads (flattened structure)
        for element_idx in range(self.num_elements):
            for param_idx in range(outdim):
                head = torch.nn.Sequential(
                    EquivariantResidualBlock(input_irreps, hidden_irreps),
                    EquivariantResidualBlock(hidden_irreps, hidden_irreps),
                    EquivariantResidualBlock(hidden_irreps, hidden_irreps),
                    o3.Linear(hidden_irreps, output_irreps, internal_weights=True, shared_weights=True)
                )
                heads.append(head)
        
        return torch.nn.ModuleList(heads)

    def _create_element_specific_combined_heads(self, outdim, parallel_units, input_irreps):
        """Create element-specific combined output heads"""
        if outdim == 0:
            return None
        
        # Create intermediate irreps
        hidden_irreps = o3.Irreps(f"{parallel_units}x0e")
        output_irreps = o3.Irreps(f"{outdim}x0e")  # Multiple scalar outputs
        
        heads = []
        # Create one combined head per element
        for element_idx in range(self.num_elements):
            head = torch.nn.Sequential(
                EquivariantResidualBlock(input_irreps, hidden_irreps),
                EquivariantResidualBlock(hidden_irreps, hidden_irreps),
                EquivariantResidualBlock(hidden_irreps, hidden_irreps),
                o3.Linear(hidden_irreps, output_irreps, internal_weights=True, shared_weights=True)
            )
            heads.append(head)
        
        return torch.nn.ModuleList(heads)

    def _print_grad_hook(self, grad: torch.Tensor, name: str):
        # Helper function for gradient hooks
        # print(f"--- Gradient Stats for {name} ---" + (f"\n  - Norm: {grad.norm().item():.4f}\n  - Mean: {grad.mean().item():.4f}\n  - Max Abs: {grad.abs().max().item():.4f}\n  - Has NaN: {torch.isnan(grad).any().item()}\n  - Has Inf: {torch.isinf(grad).any().item()}" if grad is not None else "\n--- No gradient computed for {name} ---"))
        print(f"--- Gradient Stats ---\n  - Norm: {grad.norm().item():.4f}\n  - Mean: {grad.mean().item():.4f}\n  - Max Abs: {grad.abs().max().item():.4f}\n  - Has NaN: {torch.isnan(grad).any().item()}\n  - Has Inf: {torch.isinf(grad).any().item()}")









    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        output_indices = None,
        save_intermediate: bool = False,
        save_dir: str = "./tsne_features",
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup and gradient requirements
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        
        
        # print num_elements and element_to_idx and node_attrs
        print(f"num_elements: {self.num_elements}")
        print(f"element_to_idx: {self.element_to_idx}")
        print(f"all atomic_numbers: {self.atomic_numbers}")

        
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        node_heads = data.get("head", torch.zeros_like(data["batch"]))[data["batch"]]
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        
        if compute_virials or compute_stress or compute_displacement:
            data["positions"], data["shifts"], displacement = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Compute atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[num_atoms_arange, node_heads]
        # e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs)

        # Compute embeddings and edge features
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)
        pair_node_energy = self.pair_repulsion_fn(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers) if hasattr(self, "pair_repulsion") else torch.zeros_like(node_e0)

        # Process interactions (this part remains equivariant)
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for i, (interaction, product, readout) in enumerate(zip(self.interactions, self.products, self.readouts)):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats, node_heads)[num_atoms_arange, node_heads])

            # if training:
                # node_feats_list[-1].register_hook(lambda grad: self._print_grad_hook(grad, f"equivariant_node_feats_list[-1]"))
                # node_es_list[-1].register_hook(lambda grad: self._print_grad_hook(grad, f"equivariant_node_es_list[-1]"))

            if save_intermediate:
                os.makedirs(save_dir, exist_ok=True)
                if self.elemnet:
                    np.save(f"{save_dir}/layer_{i}_equivariant_elemnet_node_feats.npy", node_feats.detach().cpu().numpy())
                elif self.globnet:
                    np.save(f"{save_dir}/layer_{i}_equivariant_globnet_node_feats.npy", node_feats.detach().cpu().numpy())
                elif self.pairnet:
                    np.save(f"{save_dir}/layer_{i}_equivariant_pairnet_node_feats.npy", node_feats.detach().cpu().numpy())

        # Use the last node features (these should be equivariant)
        node_feats_out = node_feats_list[-1]
        
        # Project to scalar features using predefined projection
        # Extract scalar features from MACE output and project to our predefined dimension
        scalar_features = self._extract_and_project_scalars(node_feats_out)
        
        # if training and scalar_features.requires_grad:
        #     scalar_features.register_hook(lambda grad: self._print_grad_hook(grad, "equivariant_scalar_features"))
        
        # Predict scale and shift (using scalar features only)
        # scale = self.scale_predictor(scalar_features) if self.scale_predictor is not None else None
        # shift = self.shift_predictor(scalar_features) if self.shift_predictor is not None else None

        #### Compute element parameters #### 
        if self.separate_output_heads and self.output_heads is not None:
            if self.separate_element_heads:
                # Route atoms to element-specific heads efficiently
                atomic_numbers = self.atomic_numbers[data["node_attrs"].argmax(dim=1)]
                print(f"atomic_numbers: {atomic_numbers}")
                params_pred_raw = torch.zeros(scalar_features.shape[0], self.outdim, 
                                            device=scalar_features.device, dtype=scalar_features.dtype)
                
                # Group atoms by element for efficient processing
                for element_z, element_idx in self.element_to_idx.items():
                    element_mask = (atomic_numbers == element_z)
                    
                    if element_mask.any():
                        element_features = scalar_features[element_mask]
                        
                        # Apply element-specific heads for all atoms of this element
                        element_outputs = []
                        for param_idx in range(self.outdim):
                            head_idx = element_idx * self.outdim + param_idx
                            param_output = self.output_heads[head_idx](element_features)
                            element_outputs.append(param_output)
                        
                        # Combine outputs and assign back
                        element_params = torch.cat(element_outputs, dim=1)
                        params_pred_raw[element_mask] = element_params
            else:
                outputs = [head(scalar_features) for head in self.output_heads]
                params_pred_raw = torch.cat(outputs, dim=1) if outputs else None
        else:
            if self.separate_element_heads and self.output_head is not None:
                # Route atoms to element-specific combined heads efficiently
                atomic_numbers = self.atomic_numbers[data["node_attrs"].argmax(dim=1)]
                params_pred_raw = torch.zeros(scalar_features.shape[0], self.outdim, 
                                            device=scalar_features.device, dtype=scalar_features.dtype)
                
                # Group atoms by element for efficient processing
                for element_z, element_idx in self.element_to_idx.items():
                    element_mask = (atomic_numbers == element_z)
                    if element_mask.any():
                        element_features = scalar_features[element_mask]
                        element_output = self.output_head[element_idx](element_features)
                        params_pred_raw[element_mask] = element_output
            else:
                params_pred_raw = self.output_head(scalar_features) if self.output_head is not None else None

        if training and params_pred_raw is not None and params_pred_raw.requires_grad:  # debug hook
            params_pred_raw.register_hook(lambda grad: self._print_grad_hook(grad, "equivariant_params_pred_raw"))
            
        # save params_pred_raw
        if save_intermediate and params_pred_raw is not None:
            np.save(f"{save_dir}/equivariant_params_pred_raw.npy", params_pred_raw.detach().cpu().numpy())

        # # Apply range constraints
        params_pred = params_pred_raw
        if params_pred_raw is not None:
            if self.use_custom_ranges:
                # Apply custom ranges (same logic as original)
                atomic_numbers = self.atomic_numbers[data["node_attrs"].argmax(dim=1)]
                deltas = torch.zeros_like(params_pred_raw)
                for i, atom in enumerate(atomic_numbers):
                    atom_num = atom.item()
                    atom_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}.get(atom_num)
                    if atom_symbol in elem_param_delta:
                        atom_deltas = torch.tensor(elem_param_delta[atom_symbol], device=params_pred_raw.device)
                        for j, param in enumerate(elem_param_included):
                            if param in ele_param_enum:
                                param_idx = ele_param_enum.index(param)
                                if param_idx < params_pred_raw.shape[1]:
                                    deltas[i, param_idx] = atom_deltas[j]
                    else:
                        raise ValueError(f'atom_symbol: {atom_symbol} not in elem_param_delta')
                
                included_mask = torch.zeros_like(params_pred_raw, dtype=torch.bool)
                for param in elem_param_included:
                    if param in ele_param_enum:
                        param_idx = ele_param_enum.index(param)
                        if param_idx < params_pred_raw.shape[1]:
                            included_mask[:, param_idx] = True
                
                # Apply element-specific ranges for custom ranges case
                if self.separate_element_heads:
                    atomic_numbers = self.atomic_numbers[data["node_attrs"].argmax(dim=1)]
                    element_ranges = torch.zeros_like(params_pred_raw)
                    for i, atom_z in enumerate(atomic_numbers):
                        element_idx = self.element_to_idx[atom_z.item()]
                        element_ranges[i] = self.half_range_pt[element_idx]
                    
                    params_pred = torch.where(
                        included_mask,
                        torch.nn.functional.tanh(params_pred_raw * 0.1) * deltas,
                        params_pred_raw * element_ranges
                    )
                else:
                    params_pred = torch.where(
                        included_mask,
                        torch.nn.functional.tanh(params_pred_raw * 0.1) * deltas,
                        params_pred_raw * self.half_range_pt
                    )
            else:
                # Apply element-specific ranges for standard case
                if self.separate_element_heads:
                    atomic_numbers = self.atomic_numbers[data["node_attrs"].argmax(dim=1)]
                    element_ranges = torch.zeros_like(params_pred_raw)
                    for i, atom_z in enumerate(atomic_numbers):
                        element_idx = self.element_to_idx[atom_z.item()]
                        element_ranges[i] = self.half_range_pt[element_idx]
                    params_pred = self.out(params_pred) * element_ranges
                else:
                    params_pred = self.out(params_pred) * self.half_range_pt

        if output_indices is not None and params_pred is not None:
            params_pred = params_pred[output_indices]

        #### Compute global parameters using equivariant projection + pooling #### 
        outputs_globpar = None
        if self.globnet:
            # Use equivariant projection on scalar features, then scatter_mean for pooling
            # This maintains equivariance because scalars are rotation invariant
            
            # Project scalar features to global scalar features
            if self.global_projection is not None:
                node_global_feats = self.global_projection(scalar_features)
            else:
                node_global_feats = scalar_features
            
            # Pool over nodes in each graph to get graph-level features
            if self.scatter_method == "scatter_mean":
                global_feats = scatter_mean(src=node_global_feats, index=data["batch"], dim=0, dim_size=num_graphs)
            else:  # scatter_sum
                global_feats = scatter_sum(src=node_global_feats, index=data["batch"], dim=0, dim_size=num_graphs)

            # scale_globpar = self.scale_predictor_globpar(global_feats) if self.scale_predictor_globpar is not None else None
            # shift_globpar = self.shift_predictor_globpar(global_feats) if self.shift_predictor_globpar is not None else None

            if self.separate_output_heads and self.output_globpar_heads is not None:
                outputs_globpar_list = [head(global_feats) for head in self.output_globpar_heads]
                outputs_globpar_raw = torch.cat(outputs_globpar_list, dim=1) if outputs_globpar_list else None
            else:
                outputs_globpar_raw = self.output_globpar_head(global_feats) if self.output_globpar_head is not None else None

            if training and outputs_globpar_raw is not None and outputs_globpar_raw.requires_grad: # debug hook
                outputs_globpar_raw.register_hook(lambda grad: self._print_grad_hook(grad, "equivariant_outputs_globpar_raw"))

            if outputs_globpar_raw is not None:
                outputs_globpar = self.out(outputs_globpar_raw) * self.half_range_pt_globpar

        #### Compute pair parameters (if needed) #### 
        pair_params = []
        if self.outdim_pair > 0:
            for graph_idx in range(num_graphs):
                start_idx = data['ptr'][graph_idx].item()
                end_idx = data['ptr'][graph_idx + 1].item()
                num_atoms = end_idx - start_idx
                graph_scalar_feats = scalar_features[start_idx:end_idx]
                
                # Create pairwise features using scalar operations (maintains equivariance)
                pairwise_feats = graph_scalar_feats.unsqueeze(0).expand(num_atoms, -1, -1) + graph_scalar_feats.unsqueeze(1).expand(-1, num_atoms, -1)
                pairwise_feats = pairwise_feats.reshape(num_atoms * num_atoms, -1)

                if self.separate_output_heads and self.output_pair_heads is not None:
                    graph_pair_params = [head(pairwise_feats) for head in self.output_pair_heads]
                    graph_pair_params = torch.cat(graph_pair_params, dim=1)
                else:
                    graph_pair_params = self.output_pair_head(pairwise_feats) if self.output_pair_head is not None else torch.zeros(num_atoms * num_atoms, self.outdim_pair, device=pairwise_feats.device)

                graph_pair_params = self.out(graph_pair_params) * self.half_range_pt_pair
                
                pair_params.append(graph_pair_params)
            
            if pair_params:
                pair_params = torch.stack(pair_params, dim=0)
                if pair_params.ndim == 2:
                    pair_params = pair_params.unsqueeze(1)
            else:
                pair_params = None



        # Prepare output dictionary
        output = {
            "params_pred": params_pred,
            "globpars_pred": outputs_globpar,
            "node_feats": scalar_features,  # Return scalar features
            "pair_param": pair_params,
        }
        
        
        
        # print(f"output: {output}")

        return output



@compile_mode("script")
class ElementSpecificEquivariantMACExTB(torch.nn.Module):
    """
    Element-specific variant of EquivariantScaleShiftMACExTB.
    Uses separate complete models for each element instead of just separate heads.
    
    WARNING: This is an experimental implementation with known limitations:
    - Pair parameters are not implemented
    - May have efficiency issues with large systems
    - Global parameter combination needs validation
    """
    
    def __init__(
        self,
        element_configs: Dict[int, Dict],  # atomic_number -> model config
        shared_config: Dict,  # shared configuration for all element models
        **kwargs,
    ):
        super().__init__()
        
        # Store element information
        self.element_configs = element_configs
        self.supported_elements = list(element_configs.keys())
        self.num_supported_elements = len(self.supported_elements)
        
        # Create element to index mapping
        self.element_to_idx = {z: i for i, z in enumerate(self.supported_elements)}
        
        # Register supported elements as buffer
        self.register_buffer(
            "supported_atomic_numbers", 
            torch.tensor(self.supported_elements, dtype=torch.int64)
        )
        
        # Store the original atomic numbers from shared config for mapping
        if "atomic_numbers" in shared_config:
            # Convert to tensor if it's a list
            if isinstance(shared_config["atomic_numbers"], list):
                original_atomic_numbers = torch.tensor(shared_config["atomic_numbers"], dtype=torch.int64)
            else:
                original_atomic_numbers = shared_config["atomic_numbers"]
            self.register_buffer(
                "original_atomic_numbers",
                original_atomic_numbers
            )
        else:
            self.register_buffer(
                "original_atomic_numbers",
                torch.tensor(self.supported_elements, dtype=torch.int64)
            )
        
        # Create separate models for each element
        self.element_models = torch.nn.ModuleDict()
        
        if not self.supported_elements:
            raise ValueError("No elements provided in element_configs")
        
        for atomic_number in self.supported_elements:
            print(f"Creating model for element {atomic_number}")
            
            # Merge shared config with element-specific config
            element_config = {**shared_config, **element_configs[atomic_number]}
            
            # Keep the original atomic_numbers and num_elements for architecture compatibility
            # Each element model will handle all element types but process only one at a time
            # This way the node_attrs shape remains consistent with the original model
            # We'll filter the data to contain only one element type before feeding to each model
            
            # Create the element-specific model with original architecture
            model_name = f"element_{atomic_number}"
            self.element_models[model_name] = EquivariantScaleShiftMACExTB(**element_config)
        
        # Store output dimensions from the first model (assuming all have same output dims)
        if not self.element_models:
            raise ValueError("No element models were created")
            
        first_model = next(iter(self.element_models.values()))
        self.outdim = first_model.outdim
        self.outdim_globpar = first_model.outdim_globpar
        self.outdim_pair = first_model.outdim_pair
        
        # Network flags
        self.elemnet = first_model.elemnet
        self.globnet = first_model.globnet
        self.pairnet = first_model.pairnet
        
        # Add compatibility attributes for training infrastructure
        # Create a combined ModuleList that contains all interactions from all element models
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        
        for model in self.element_models.values():
            self.interactions.extend(model.interactions)
            self.products.extend(model.products)
            self.readouts.extend(model.readouts)
        
        # Delegate other attributes to the first model for compatibility
        self.node_embedding = first_model.node_embedding
        self.radial_embedding = first_model.radial_embedding
        self.spherical_harmonics = first_model.spherical_harmonics
        self.atomic_energies_fn = first_model.atomic_energies_fn
        
        # Add other commonly accessed attributes
        self.r_max = first_model.r_max
        self.num_interactions = first_model.num_interactions
        # FIX: Use original atomic numbers instead of first model's atomic numbers
        self.atomic_numbers = self.original_atomic_numbers
        
    def _group_atoms_by_element(self, data: Dict[str, torch.Tensor]) -> tuple:
        """Group atoms by their element type and create separate data dictionaries"""
        
        # Get atomic numbers for each atom from one-hot encoding
        atomic_numbers_onehot = data["node_attrs"].argmax(dim=1)
        
        # Use the stored original atomic numbers for mapping
        original_atomic_numbers = self.original_atomic_numbers
        
        grouped_data = {}
        atom_indices_map = {}  # Track original indices for result reconstruction
        
        for element_z in self.supported_elements:
            # Find the index of this element in the original encoding
            element_mask_list = (original_atomic_numbers == element_z)
            if not element_mask_list.any():
                continue  # This element not in original model
                
            element_idx_in_original = torch.where(element_mask_list)[0][0].item()
                
            # Find atoms of this element
            element_mask = (atomic_numbers_onehot == element_idx_in_original)
            
            if not element_mask.any():
                continue  # Skip if no atoms of this element
                
            element_indices = torch.where(element_mask)[0]
            atom_indices_map[element_z] = element_indices
            
            # Create data subset for this element
            element_data = {}
            
            # Node-level data - create proper element encoding
            # Ensure consistent device and dtype
            device = data["node_attrs"].device
            dtype = data["node_attrs"].dtype
            num_elements = data["node_attrs"].shape[1]  # Get original number of elements
            
            # GRADIENT-SAFE: Use original data slicing to preserve gradients
            # Extract only atoms of this element type while preserving gradients
            element_data["node_attrs"] = data["node_attrs"][element_indices]  # Preserves gradients
            element_data["positions"] = data["positions"][element_indices]  # Preserves gradients
            
            # Edge-level data - filter and reindex edges
            edge_mask = torch.isin(data["edge_index"][0], element_indices) & torch.isin(data["edge_index"][1], element_indices)
            
            if edge_mask.any():
                element_edges = data["edge_index"][:, edge_mask]
                
                # Safe edge reindexing - use dictionary mapping approach
                old_to_new_idx = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(element_indices)}
                
                # Create reindexed edges safely with proper data type
                element_edges_reindexed = torch.zeros_like(element_edges, dtype=torch.long, device=device)
                for i in range(element_edges.shape[1]):
                    src_idx = element_edges[0, i].item()
                    dst_idx = element_edges[1, i].item()
                    
                    # Ensure indices exist in mapping (they should due to edge_mask filtering)
                    if src_idx in old_to_new_idx and dst_idx in old_to_new_idx:
                        new_src = old_to_new_idx[src_idx]
                        new_dst = old_to_new_idx[dst_idx]
                        
                        # Validate indices are within bounds
                        if 0 <= new_src < len(element_indices) and 0 <= new_dst < len(element_indices):
                            element_edges_reindexed[0, i] = new_src
                            element_edges_reindexed[1, i] = new_dst
                        else:
                            print(f"Warning: Reindexed edge out of bounds: ({new_src}, {new_dst}), max={len(element_indices)-1}")
                            element_edges_reindexed[0, i] = 0
                            element_edges_reindexed[1, i] = 0
                    else:
                        # This shouldn't happen due to filtering, but safety check
                        print(f"Warning: Edge index not found in mapping: {src_idx}, {dst_idx}")
                        element_edges_reindexed[0, i] = 0  # Safe fallback
                        element_edges_reindexed[1, i] = 0
                
                element_data["edge_index"] = element_edges_reindexed
                
                # GRADIENT-SAFE: Handle shifts preserving gradients
                if "shifts" in data:
                    element_data["shifts"] = data["shifts"][edge_mask]  # Preserves gradients
                else:
                    element_data["shifts"] = torch.zeros(element_edges.shape[1], 3, device=device, dtype=data["positions"].dtype)
            else:
                # No edges for this element (isolated atoms)
                element_data["edge_index"] = torch.zeros((2, 0), dtype=torch.long, device=device)
                element_data["shifts"] = torch.zeros((0, 3), device=device, dtype=data["positions"].dtype)
            
            # Graph-level data - treat as single graph with consistent device/dtype
            element_data["batch"] = torch.zeros(len(element_indices), dtype=torch.long, device=device)
            element_data["ptr"] = torch.tensor([0, len(element_indices)], dtype=torch.long, device=device)
            
            # GRADIENT-SAFE: Copy other necessary data preserving gradients
            for key in ["cell", "unit_shifts"]:
                if key in data:
                    # These are typically constant tensors that don't need gradients, but preserve them anyway
                    element_data[key] = data[key] if isinstance(data[key], torch.Tensor) else data[key]
            
            # Handle head information - create consistent head data for element subset
            # Since each element model has only one element type, set all heads to 0
            if "head" in data:
                element_data["head"] = torch.zeros(len(element_indices), dtype=data["head"].dtype, device=device)
            else:
                element_data["head"] = torch.zeros(len(element_indices), dtype=torch.long, device=device)
            
            grouped_data[element_z] = element_data
            
        return grouped_data, atom_indices_map
    
    def _combine_element_outputs(
        self, 
        element_outputs: Dict[int, Dict[str, torch.Tensor]], 
        atom_indices_map: Dict[int, torch.Tensor],
        total_atoms: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Combine outputs from different element models"""
        
        combined_output = {}
        
        # GRADIENT-SAFE: Handle params_pred (node-level) preserving gradients
        if self.elemnet and any("params_pred" in output and output["params_pred"] is not None for output in element_outputs.values()):
            # Get dtype from first available output to ensure consistency
            first_output = next(output["params_pred"] for output in element_outputs.values() if "params_pred" in output and output["params_pred"] is not None)
            combined_params = torch.zeros((total_atoms, self.outdim), device=device, dtype=first_output.dtype)
            
            for element_z, output in element_outputs.items():
                if "params_pred" in output and output["params_pred"] is not None:
                    original_indices = atom_indices_map[element_z]
                    # In-place assignment preserves gradients from element outputs
                    combined_params[original_indices] = output["params_pred"]
            combined_output["params_pred"] = combined_params
        else:
            combined_output["params_pred"] = None
            
        # FIX: Global parameters combination - use mean instead of sum for graph-level properties
        if self.globnet:
            globpars_list = []
            element_weights = []  # Weight by number of atoms per element
            
            for element_z, output in element_outputs.items():
                if "globpars_pred" in output and output["globpars_pred"] is not None:
                    globpars_list.append(output["globpars_pred"])
                    # Weight by number of atoms of this element
                    num_atoms_element = len(atom_indices_map[element_z])
                    element_weights.append(num_atoms_element)
            
            if globpars_list:
                # GRADIENT-SAFE: Weighted average preserving gradients
                globpars_tensor = torch.stack(globpars_list)  # Preserves gradients
                weights_tensor = torch.tensor(element_weights, device=device, dtype=globpars_tensor.dtype)
                weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize weights
                
                # This operation preserves gradients through globpars_tensor
                combined_output["globpars_pred"] = torch.sum(
                    globpars_tensor * weights_tensor.unsqueeze(-1), dim=0
                )
            else:
                combined_output["globpars_pred"] = None
        else:
            combined_output["globpars_pred"] = None
            
        # GRADIENT-SAFE: Handle node_feats (node-level) preserving gradients
        if any("node_feats" in output and output["node_feats"] is not None for output in element_outputs.values()):
            # Get feature dimension and dtype from first available output
            feat_dim = None
            first_node_feats = None
            for output in element_outputs.values():
                if "node_feats" in output and output["node_feats"] is not None:
                    feat_dim = output["node_feats"].shape[-1]
                    first_node_feats = output["node_feats"]
                    break
            
            if feat_dim is not None and first_node_feats is not None:
                combined_feats = torch.zeros((total_atoms, feat_dim), device=device, dtype=first_node_feats.dtype)
                for element_z, output in element_outputs.items():
                    if "node_feats" in output and output["node_feats"] is not None:
                        original_indices = atom_indices_map[element_z]
                        # In-place assignment preserves gradients from element outputs
                        combined_feats[original_indices] = output["node_feats"]
                combined_output["node_feats"] = combined_feats
            else:
                combined_output["node_feats"] = None
        else:
            combined_output["node_feats"] = None
            
        # WARNING: Pair parameters not implemented - this is a known limitation
        combined_output["pair_param"] = None
        
        return combined_output
    
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        output_indices = None,
        save_intermediate: bool = False,
        save_dir: str = "./tsne_features",
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        # Debug: Check input data integrity before processing
        try:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print(f"Warning: Invalid values in input {key}: NaN={torch.isnan(value).any()}, Inf={torch.isinf(value).any()}")
                    if value.dtype == torch.long and key in ["edge_index", "batch"] and (value < 0).any():
                        print(f"Warning: Negative indices in input {key}: min={value.min()}, max={value.max()}")
        except Exception as e:
            print(f"Debug check failed: {e}")
        
        # Group atoms by element
        grouped_data, atom_indices_map = self._group_atoms_by_element(data)
        
        # Check if we have any atoms to process
        if not grouped_data:
            # Return empty results if no supported elements found
            device = data["positions"].device
            total_atoms = data["positions"].shape[0]
            return {
                "params_pred": torch.zeros((total_atoms, self.outdim), device=device) if self.elemnet else None,
                "globpars_pred": torch.zeros((1, self.outdim_globpar), device=device) if self.globnet else None,
                "node_feats": torch.zeros((total_atoms, 64), device=device),  # Default feature size
                "pair_param": None,
            }
        
        # Process each element with its dedicated model
        element_outputs = {}
        for element_z, element_data in grouped_data.items():
            model_name = f"element_{element_z}"
            if model_name not in self.element_models:
                continue  # Skip if model doesn't exist for this element
                
            model = self.element_models[model_name]
            
            # Debug: Check element data integrity before forward pass
            try:
                for key, value in element_data.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            print(f"Warning: Invalid values in element_{element_z} {key}: NaN={torch.isnan(value).any()}, Inf={torch.isinf(value).any()}")
                        if value.dtype == torch.long and key == "edge_index" and value.numel() > 0:
                            max_node_idx = element_data["positions"].shape[0] - 1
                            if (value < 0).any() or (value > max_node_idx).any():
                                print(f"Warning: Invalid edge indices in element_{element_z}: min={value.min()}, max={value.max()}, expected range=[0,{max_node_idx}]")
            except Exception as e:
                print(f"Element data debug check failed for element_{element_z}: {e}")
            
            # Forward pass for this element
            element_output = model(
                element_data,
                training=training,
                compute_force=compute_force,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
                compute_displacement=compute_displacement,
                compute_hessian=compute_hessian,
                output_indices=None,  # Will handle output_indices at the end
                save_intermediate=save_intermediate,
                save_dir=f"{save_dir}/element_{element_z}" if save_intermediate else save_dir,
            )
            
            element_outputs[element_z] = element_output
        
        # Combine outputs from all elements
        total_atoms = data["positions"].shape[0]
        combined_output = self._combine_element_outputs(
            element_outputs, 
            atom_indices_map, 
            total_atoms, 
            data["positions"].device
        )
        
        # Apply output_indices if specified
        if output_indices is not None and combined_output["params_pred"] is not None:
            combined_output["params_pred"] = combined_output["params_pred"][output_indices]
        
        # GRADIENT VERIFICATION: Check that gradients are preserved
        if training and hasattr(data["positions"], 'requires_grad') and data["positions"].requires_grad:
            for key, value in combined_output.items():
                if value is not None and isinstance(value, torch.Tensor) and value.numel() > 0:
                    if not value.requires_grad:
                        print(f"Warning: Output {key} does not require gradients despite input requiring gradients")
                    # else:
                    #     print(f"✓ Output {key} properly requires gradients")
            
        return combined_output