import sys
import mace.cli.run_train as run_train

args = [
    "--name", "MACE_model",
    "--train_file", "BOTNet-datasets/dataset_3BPA/train_300K.xyz",
    "--valid_fraction", "0.05",
    "--forces_key", "forces",
    "--energy_key", "energy",
    "--test_file", "BOTNet-datasets/dataset_3BPA/test_300K.xyz",
    "--E0s", "{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}",
    "--model", "ScaleShiftMACE",
    "--hidden_irreps", "32x0e",
    "--r_max", "4.0",
    "--batch_size", "20",
    "--max_num_epochs", "3",
    "--ema",
    "--ema_decay", "0.99",
    "--amsgrad",
    "--default_dtype", "float32",
    "--device", "cpu",
    "--seed", "123",
    "--swa"
]

sys.argv = [sys.argv[0]] + args
run_train.main()



# from ase.io import read
# import numpy as np
# from mace.calculators import MACECalculator


# calculator = MACECalculator(model_paths='/content/checkpoints/MACE_model_run-123.model', device='cuda')
# init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
# descriptors = calculator.get_descriptors(init_conf)