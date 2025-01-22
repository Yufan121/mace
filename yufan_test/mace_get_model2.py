import sys
import mace.cli.run_train as run_train
from mace import data, tools, modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

def train_model(model, train_loader, val_loaders, device, args, max_num_epochs=10):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = modules.WeightedEnergyForcesLoss(
        energy_weight=0.1, forces_weight=0.9
    )
    
    for epoch in range(max_num_epochs):
        
        for param in model.parameters():
            param.requires_grad = False
        # Validation
        model.eval()
        total_val_loss = 0.0
        # with torch.no_grad():     # This is not needed
        for head, val_loader in val_loaders.items():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.to_dict(), training=False, compute_force=True)
                print(f"outputs: {outputs}")
                # loss = loss_fn(pred=outputs, ref=batch)
                # total_val_loss += loss.item()

        avg_val_loss = total_val_loss / sum(len(loader) for loader in val_loaders.values())
        print(f"Epoch {epoch + 1}/{max_num_epochs}, Validation Loss: {avg_val_loss}")

        for param in model.parameters():
            param.requires_grad = True

        # Train
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.to_dict(), training=True, compute_force=True)
            # print(f'outputs: {outputs}')
            loss = loss_fn(pred=outputs, ref=batch)  
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{max_num_epochs}, Train Loss: {avg_train_loss}")









if __name__ == "__main__":
    args = [
        "--name", "MACE_model",
        "--train_file", "BOTNet-datasets/dataset_3BPA/train_300K.xyz",
        "--valid_fraction", "0.05",
        "--forces_key", "forces",
        "--energy_key", "energy",
        "--test_file", "BOTNet-datasets/dataset_3BPA/test_300K.xyz",
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
    print(model, train_loader, val_loaders, test_loader)



    ### Test run with loader

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train the model
    train_model(model, train_loader, val_loaders, device, args, 5)



