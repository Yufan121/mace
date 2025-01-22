import torch
from e3nn import o3
from mace.modules.models import ScaleShiftMACE
from mace import modules
import ast
from torch_geometric.data import Data, DataLoader
from typing import List
from torch.utils.data import ConcatDataset
import torch_geometric
from mace.calculators.foundations_models import mace_mp, mace_off
import logging
from mace.tools.multihead_tools import (
    HeadConfig,
    assemble_mp_data,
    dict_head_to_dataclass,
    prepare_default_head,
)


# def prepare_model_foundation(args):
#     model_foundation = None
#     if args.foundation_model is not None:
#         if args.foundation_model in ["small", "medium", "large"]:
#             logging.info(f"Using foundation model mace-mp-0 {args.foundation_model} as initial checkpoint.")
#             calc = mace_mp(model=args.foundation_model, device=args.device, default_dtype=args.default_dtype)
#             model_foundation = calc.models[0]
#         elif args.foundation_model in ["small_off", "medium_off", "large_off"]:
#             model_type = args.foundation_model.split("_")[0]
#             logging.info(f"Using foundation model mace-off-2023 {model_type} as initial checkpoint. ASL license.")
#             calc = mace_off(model=model_type, device=args.device, default_dtype=args.default_dtype)
#             model_foundation = calc.models[0]
#         else:
#             model_foundation = torch.load(args.foundation_model, map_location=args.device)
#             logging.info(f"Using foundation model {args.foundation_model} as initial checkpoint.")
#         args.r_max = model_foundation.r_max.item()
#     else:
#         args.multiheads_finetuning = False
#     return model_foundation


# def prepare_data_loaders(args, z_table, model_foundation=None):
#     if args.get("heads") is None:
#         args["heads"] = prepare_default_head(args)
#     logging.info("===========LOADING INPUT DATA===========")
#     heads = list(args["heads"])
#     logging.info(f"Using heads: {heads}")
#     head_configs: List[HeadConfig] = []
#     for head, head_args in args["heads"].items():
#         logging.info(f"============= Processing head {head} ===========")
#         head_config = dict_head_to_dataclass(head_args, head, args)
#         if head_config.statistics_file is not None:
#             with open(head_config.statistics_file, "r") as f:
#                 statistics = json.load(f)
#             logging.info("Using statistics json file")
#             head_config.r_max = (statistics["r_max"] if args.get("foundation_model") is None else args["r_max"])
#             head_config.atomic_numbers = statistics["atomic_numbers"]
#             head_config.mean = statistics["mean"]
#             head_config.std = statistics["std"]
#             head_config.avg_num_neighbors = statistics["avg_num_neighbors"]
#             head_config.compute_avg_num_neighbors = False
#             if isinstance(statistics["atomic_energies"], str) and statistics["atomic_energies"].endswith(".json"):
#                 with open(statistics["atomic_energies"], "r", encoding="utf-8") as f:
#                     atomic_energies = json.load(f)
#                 head_config.E0s = atomic_energies
#                 head_config.atomic_energies_dict = ast.literal_eval(atomic_energies)
#             else:
#                 head_config.E0s = statistics["atomic_energies"]
#                 head_config.atomic_energies_dict = ast.literal_eval(statistics["atomic_energies"])
#         # Data preparation
#         if check_path_ase_read(head_config.train_file):
#             if head_config.valid_file is not None:
#                 assert check_path_ase_read(head_config.valid_file), "valid_file if given must be same format as train_file"
#             config_type_weights = get_config_type_weights(head_config.config_type_weights)
#             collections, atomic_energies_dict = get_dataset_from_xyz(
#                 work_dir=args["work_dir"],
#                 train_path=head_config.train_file,
#                 valid_path=head_config.valid_file,
#                 valid_fraction=head_config.valid_fraction,
#                 config_type_weights=config_type_weights,
#                 test_path=head_config.test_file,
#                 seed=args["seed"],
#                 energy_key=head_config.energy_key,
#                 forces_key=head_config.forces_key,
#                 stress_key=head_config.stress_key,
#                 virials_key=head_config.virials_key,
#                 dipole_key=head_config.dipole_key,
#                 charges_key=head_config.charges_key,
#                 head_name=head_config.head_name,
#                 keep_isolated_atoms=head_config.keep_isolated_atoms,
#             )
#             head_config.collections = collections
#             head_config.atomic_energies_dict = atomic_energies_dict
#             logging.info(
#                 f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
#                 f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],"
#             )
#             head_configs.append(head_config)
#     if all(check_path_ase_read(head_config.train_file) for head_config in head_configs):
#         size_collections_train = sum(len(head_config.collections.train) for head_config in head_configs)
#         size_collections_valid = sum(len(head_config.collections.valid) for head_config in head_configs)
#         if size_collections_train < args["batch_size"]:
#             logging.error(f"Batch size ({args['batch_size']}) is larger than the number of training data ({size_collections_train})")
#         if size_collections_valid < args["valid_batch_size"]:
#             logging.warning(f"Validation batch size ({args['valid_batch_size']}) is larger than the number of validation data ({size_collections_valid})")
#     train_sets = {head: [] for head in heads}
#     valid_sets = {head: [] for head in heads}
#     for head_config in head_configs:
#         if check_path_ase_read(head_config.train_file):
#             train_sets[head_config.head_name] = [
#                 data.AtomicData.from_config(config, z_table=z_table, cutoff=args["r_max"], heads=heads)
#                 for config in head_config.collections.train
#             ]
#             valid_sets[head_config.head_name] = [
#                 data.AtomicData.from_config(config, z_table=z_table, cutoff=args["r_max"], heads=heads)
#                 for config in head_config.collections.valid
#             ]
#         elif head_config.train_file.endswith(".h5"):
#             train_sets[head_config.head_name] = data.HDF5Dataset(
#                 head_config.train_file, r_max=args["r_max"], z_table=z_table, heads=heads, head=head_config.head_name
#             )
#             valid_sets[head_config.head_name] = data.HDF5Dataset(
#                 head_config.valid_file, r_max=args["r_max"], z_table=z_table, heads=heads, head=head_config.head_name
#             )
#         else:
#             train_sets[head_config.head_name] = data.dataset_from_sharded_hdf5(
#                 head_config.train_file, r_max=args["r_max"], z_table=z_table, heads=heads, head=head_config.head_name
#             )
#             valid_sets[head_config.head_name] = data.dataset_from_sharded_hdf5(
#                 head_config.valid_file, r_max=args["r_max"], z_table=z_table, heads=heads, head=head_config.head_name
#             )
#         train_loader_head = torch_geometric.dataloader.DataLoader(
#             dataset=train_sets[head_config.head_name],
#             batch_size=args["batch_size"],
#             shuffle=True,
#             drop_last=True,
#             pin_memory=args["pin_memory"],
#             num_workers=args["num_workers"],
#             generator=torch.Generator().manual_seed(args["seed"]),
#         )
#         head_config.train_loader = train_loader_head
#     train_set = ConcatDataset([train_sets[head] for head in heads])
#     train_sampler, valid_sampler = None, None
#     if args["distributed"]:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(
#             train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args["seed"]
#         )
#         valid_samplers = {}
#         for head, valid_set in valid_sets.items():
#             valid_sampler = torch.utils.data.distributed.DistributedSampler(
#                 valid_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args["seed"]
#             )
#             valid_samplers[head] = valid_sampler
#     train_loader = torch_geometric.dataloader.DataLoader(
#         dataset=train_set,
#         batch_size=args["batch_size"],
#         sampler=train_sampler,
#         shuffle=(train_sampler is None),
#         drop_last=(train_sampler is None),
#         pin_memory=args["pin_memory"],
#         num_workers=args["num_workers"],
#         generator=torch.Generator().manual_seed(args["seed"]),
#     )
#     valid_loaders = {heads[i]: None for i in range(len(heads))}
#     if not isinstance(valid_sets, dict):
#         valid_sets = {"Default": valid_sets}
#     for head, valid_set in valid_sets.items():
#         valid_loaders[head] = torch_geometric.dataloader.DataLoader(
#             dataset=valid_set,
#             batch_size=args["valid_batch_size"],
#             sampler=valid_samplers[head] if args["distributed"] else None,
#             shuffle=False,
#             drop_last=False,
#             pin_memory=args["pin_memory"],
#             num_workers=args["num_workers"],
#             generator=torch.Generator().manual_seed(args["seed"]),
#         )
#     return train_loader, valid_loaders


if __name__ == "__main__":
    # Define the required arguments
    args = {
        "forces": True,
        
        "r_max": 4.0,
        "num_radial_basis": 8,
        "num_cutoff_basis": 6,
        "max_ell": 3,
        "interaction": "RealAgnosticResidualInteractionBlock",  # Replace with the actual interaction block class name
        "interaction_first": "RealAgnosticResidualInteractionBlock",  # Replace with the actual interaction block class name
        "num_interactions": 3,
        "num_elements": 4,
        "hidden_irreps": "32x0e",
        "MLP_irreps": "32x0e",
        "atomic_energies": torch.tensor([-13.663181292231226, -1029.2809654211628, -1484.1187695035828, -2042.0330099956639]),
        "avg_num_neighbors": 16.0,
        "atomic_numbers": [1, 6, 7, 8],
        "correlation": 2,
        "gate": "None",  # Replace with the actual gate function name
        "pair_repulsion": False,
        "distance_transform": "None",
        "radial_MLP": "[64, 64, 64]",
        "radial_type": "bessel",
        "heads": ["default"],
        "std": 1.0,
        "mean": 0.0,
        "batch_size": 1
    }

    # Create an instance of ScaleShiftMACE
    model = ScaleShiftMACE(
        r_max=args["r_max"],
        num_bessel=args["num_radial_basis"],
        num_polynomial_cutoff=args["num_cutoff_basis"],
        max_ell=args["max_ell"],
        interaction_cls=modules.interaction_classes[args["interaction"]],
        interaction_cls_first=modules.interaction_classes[args["interaction_first"]],
        num_interactions=args["num_interactions"],
        num_elements=args["num_elements"],
        hidden_irreps=o3.Irreps(args["hidden_irreps"]),
        MLP_irreps=o3.Irreps(args["MLP_irreps"]),
        atomic_energies=args["atomic_energies"],
        avg_num_neighbors=args["avg_num_neighbors"],
        atomic_numbers=args["atomic_numbers"],
        correlation=args["correlation"],
        gate=modules.gate_dict[args["gate"]],
        pair_repulsion=args["pair_repulsion"],
        distance_transform=args["distance_transform"],
        atomic_inter_scale=args["std"],
        atomic_inter_shift=args["mean"],
        radial_MLP=ast.literal_eval(args["radial_MLP"]),
        radial_type=args["radial_type"],
        heads=args["heads"]
    )

    print(model)

    # print model num_parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")









    # Create a dummy dataset
    num_atoms = 10
    positions = torch.rand((num_atoms, 3))  # Random positions
    atomic_numbers = torch.randint(1, 9, (num_atoms,))  # Random atomic numbers between 1 and 8
    node_attrs = torch.rand((num_atoms, 32))  # Random node attributes
    batch = torch.zeros(num_atoms, dtype=torch.long)  # Single batch

    # Create a Data object
    data = Data(
        positions=positions,
        z=atomic_numbers,
        node_attrs=node_attrs,
        batch=batch,
        energy=torch.tensor([0.0]),  # Dummy energy
        forces=torch.zeros((num_atoms, 3))  # Dummy forces
    )

    # Create a DataLoader
    train_loader = DataLoader(
        dataset=[data],
        batch_size=args["batch_size"],
        shuffle=True,
        drop_last=True,
        # pin_memory=args["pin_memory"],
        # num_workers=args["num_workers"],
        # generator=torch.Generator().manual_seed(args["seed"])
    )




    # # model_foundation = prepare_model_foundation(args)
    # from mace.tools.utils import AtomicNumberTable 
    # # Define the atomic numbers you are working with 
    # atomic_numbers = [1, 6, 7, 8] # Example: Hydrogen, Carbon, Nitrogen, Oxygen 
    
    # # Create the AtomicNumberTable 
    # z_table = AtomicNumberTable(atomic_numbers)
    
    # train_loader, valid_loaders = prepare_data_loaders(args, z_table)

    # Evaluate the model on the dummy dataset
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to('cpu')  # Move to the appropriate device
            output = model(batch.to_dict(), training=False)
            print(output)
