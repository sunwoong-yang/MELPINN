import argparse
import torch
from training import train_pinn, save_results, plot_heights, plot_velocities
from network import PINN
from utils import install, make_or_clear_figures_directory, get_activation_function

# List of required packages
required_packages = [
    "torch",
    "numpy",
    "matplotlib",
    "tqdm"
]

# Install required packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

torch.set_num_threads(12)  # Set the number of threads for PyTorch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    print('GPU name:', gpu_name)

# Argument parsing
parser = argparse.ArgumentParser(description='Train a Physics-Informed Neural Network (PINN).')
parser.add_argument('-tag', type=str, default='', help='Directory tag.')
parser.add_argument('-e', '--epochs', type=int, default=30000, help='Number of epochs for training.')
parser.add_argument('-T', '--N_tank', type=int, default=6, help='Number of tanks.')
parser.add_argument('-N', '--N_collo', type=int, default=5000, help='Number of collocation points.')
parser.add_argument('-hl', '--hidden_layers', type=int, nargs='+', default=[64] * 7,
                    help='List of hidden layers sizes.')
parser.add_argument('-t', '--end_time', type=int, default=2758, help='End time for the simulation.')
parser.add_argument('-BC', '--use_hard_BC', type=str, default='True', help='Use hard boundary conditions (True/False).')
parser.add_argument('-sc', '--scheduler', nargs=2, type=int, default=[1000, 0.98],
                    help='List with two elements: [step, gamma].')
parser.add_argument('-w', '--loss_weight', nargs=2, type=int, default=[10, 1],
                    help='List with two elements: [mom_weight, conti_weight].')
parser.add_argument('-af', '--activation_function', type=str, default='Tanh',
                    choices=['Tanh', 'ReLU', 'Sigmoid', 'LeakyReLU'],
                    help='Activation function to use in the neural network.')

args = parser.parse_args()

# Parsing command-line arguments
directory_tag = args.tag
N_tank = args.N_tank
epochs = args.epochs
N_collo = args.N_collo
hidden_layers = args.hidden_layers
end_time = args.end_time
use_hard_BC = args.use_hard_BC.lower() in ['t', 'true']
scheduler_step, scheduler_gamma = args.scheduler[0], args.scheduler[1]
momentum_weight, continuity_weight = args.loss_weight[0], args.loss_weight[1]
activation_function = args.activation_function

# Construct the folder name based on the argument values
hidden_layers_str = 'x'.join(map(str, hidden_layers))
folder_name = (
    f"figures/[{directory_tag}]Ntanks{N_tank}-e{epochs}-N{N_collo}"
    f"-hl{hidden_layers_str}-t{end_time}-BC{use_hard_BC}"
    f"-lr[{scheduler_step}, {scheduler_gamma}]"
    f"-lossW[{momentum_weight}, {continuity_weight}]"
    f"-af[{activation_function}]"
)

# Clear or create the specified figures directory
make_or_clear_figures_directory(folder_name)

# Train and validate the model
net = train_pinn(weights=[momentum_weight] * (N_tank - 1) + [continuity_weight] * N_tank, use_hard_BC=use_hard_BC,
                 epochs=epochs, N_collo=N_collo, end_time=end_time, plot_period=1000,
                 hidden_layers=hidden_layers, activation_function=activation_function)
final_results = save_results(net, input=end_time)
plot_heights(net, t_end=end_time)
plot_velocities(net, t_end=end_time)
