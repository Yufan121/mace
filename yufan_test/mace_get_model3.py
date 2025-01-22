import sys
import mace.cli.run_train as run_train
from mace import data, tools, modules
import torch
import torch.nn.functional as F
from torch.optim import Adam
import ase
from mace.tools.utils import AtomicNumberTable
# from torch_geometric.data import Batch
# import torch_geometric
from mace.tools import torch_geometric

# def train_model(model, train_loader, val_loaders, device, args, max_num_epochs=10):
#     model.to(device)
#     optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     # loss_fn = modules.WeightedEnergyForcesLoss(
#     #     energy_weight=0.1, forces_weight=0.9
#     # )
#     loss_mse = torch.nn.MSELoss()
#     fake_target = None    
#     fake_target_trn = None    
    
#     for epoch in range(max_num_epochs):
        
#         for param in model.parameters():
#             param.requires_grad = False
#         # Validation
#         model.eval()
#         total_val_loss = 0.0
#         # with torch.no_grad():     # This is not needed
#         for head, val_loader in val_loaders.items():
#             for batch in val_loader:
#                 batch = batch.to(device)
#                 outputs = model(batch.to_dict(), training=False, compute_force=True)
#                 # print(f"VAL outputs[params_pred].shape: {outputs["params_pred"].shape}")
#                 # loss = loss_fn(pred=outputs, ref=batch)
#                 fake_target = torch.zeros_like(outputs["params_pred"]) 
#                 loss = loss_mse(outputs["params_pred"], fake_target)
#                 total_val_loss += loss.item()

#         avg_val_loss = total_val_loss / sum(len(loader) for loader in val_loaders.values())
#         print(f"Epoch {epoch + 1}/{max_num_epochs}, Validation Loss: {avg_val_loss}")

#         for param in model.parameters():
#             param.requires_grad = True

#         # Train
#         model.train()
#         total_train_loss = 0.0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch.to_dict(), training=True, compute_force=True)
#             # print(f'outputs: {outputs}')
#             # loss = loss_fn(pred=outputs, ref=batch)  
#             fake_target_trn = torch.zeros_like(outputs["params_pred"]) 
#             loss = loss_mse(outputs["params_pred"], fake_target_trn)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)
#         print(f"Epoch {epoch + 1}/{max_num_epochs}, Train Loss: {avg_train_loss}")

def train_model(model, train_loader, val_loaders, device, dataset, max_num_epochs=10):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # loss_fn = modules.WeightedEnergyForcesLoss(
    #     energy_weight=0.1, forces_weight=0.9
    # )
    loss_mse = torch.nn.MSELoss()
    fake_target = None    
    fake_target_trn = None    
    
    for epoch in range(max_num_epochs):
        
        for param in model.parameters():
            param.requires_grad = False
        # Validation
        model.eval()
        total_val_loss = 0.0
        # with torch.no_grad():     # This is not needed
        for head, val_loader in val_loaders.items():
            for batch in val_loader:
                batch_old = batch.to(device)
                
                data = dataset[0]
                batch = torch_geometric.Batch.from_data_list([data])
                
                
                outputs = model(batch.to_dict(), training=False, compute_force=True)
                print(f"VAL outputs[params_pred].shape: {outputs["params_pred"].shape}")
                # loss = loss_fn(pred=outputs, ref=batch)
                fake_target = torch.zeros_like(outputs["params_pred"]) 
                loss = loss_mse(outputs["params_pred"], fake_target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / sum(len(loader) for loader in val_loaders.values())
        print(f"Epoch {epoch + 1}/{max_num_epochs}, Validation Loss: {avg_val_loss}")

        for param in model.parameters():
            param.requires_grad = True

        # Train
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            # batch = batch.to(device)
            data = dataset[0]
            batch = torch_geometric.Batch.from_data_list([data])
            optimizer.zero_grad()
            outputs = model(batch.to_dict(), training=True, compute_force=True)
            # print(f'outputs: {outputs}')
            # loss = loss_fn(pred=outputs, ref=batch)  
            fake_target_trn = torch.zeros_like(outputs["params_pred"]) 
            loss = loss_mse(outputs["params_pred"], fake_target_trn)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{max_num_epochs}, Train Loss: {avg_train_loss}")


def cvt2datamace(list_all_z: list[int] = [8,1,6,7]):
    
    # file_path = "BOTNet-datasets/dataset_3BPA/train_300K.xyz"
    file_path = "/scratch/kx58/yx7184/githubs/NNxTB/scratchfolder_par_4_mace/data_processing/sgdml/md17_aspirin_train/frame_0.xyz"

    
    ase_list = ase.io.read(file_path, index=":")    # atom list

    configs = data.utils.config_from_atoms_list(ase_list)   # get configurations
    
    z_table = AtomicNumberTable(sorted(list_all_z))

    dataset = []
    for config in configs:
        dataset.append(data.AtomicData.from_config(config, z_table, 6.0))
    
    return dataset


def get_model_and_loaders():
    # Backup the original sys.argv
    original_argv = sys.argv.copy()

    args = [
        "--name", "MACE_model",
        "--train_file", "/scratch/kx58/yx7184/githubs/mace/yufan_test/BOTNet-datasets/dataset_3BPA/train_300K.xyz",
        "--valid_fraction", "0.05",
        "--forces_key", "forces",
        "--energy_key", "energy",
        "--test_file", "/scratch/kx58/yx7184/githubs/mace/yufan_test/BOTNet-datasets/dataset_3BPA/test_300K.xyz",
        "--E0s", "{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}",
        "--model", "ScaleShiftMACExTB",
        "--hidden_irreps", "32x0e",
        "--r_max", "4.0",
        "--batch_size", "20",
        "--max_num_epochs", "0",
        "--ema",
        "--ema_decay", "0.99",
        "--amsgrad",
        "--default_dtype", "float32",
        "--device", "cpu",
        "--seed", "123",
        "--swa",
        "--outdim", "21"
    ]

    sys.argv = [sys.argv[0]] + args
    args = tools.build_default_arg_parser().parse_args()

    # Load the model and data loaders
    model, train_loader, val_loaders, test_loader = run_train.get_model_and_loader(args)

    # Restore the original sys.argv
    sys.argv = original_argv

    return model, train_loader, val_loaders, test_loader





if __name__ == "__main__":
    
    model, train_loader, val_loaders, test_loader = get_model_and_loaders()
    print(model)


    ### Test run with loader

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get dataset
    dataset = cvt2datamace()
    # Train the model
    train_model(model, train_loader, val_loaders, device, dataset=dataset, max_num_epochs=50)



