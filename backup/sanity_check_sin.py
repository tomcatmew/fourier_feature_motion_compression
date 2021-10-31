import torch
import torch.nn as nn
import math
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

# Sanity check code where the target function is just a 1D sin curve

class Net(nn.Module):
    def __init__(self,
        in_dim=2,
        out_dim=1,
        num_hidden=1,
        hidden_width=32,
        apply_activation=False
        ):
        super(Net, self).__init__()

        layers = list()
        in_features = in_dim
        arch = [hidden_width] * num_hidden + [out_dim]
        for idx, out_features in enumerate(arch):
            layers += [nn.Linear(in_features, out_features)]
            if apply_activation and idx < len(arch) - 1: # Do not add activation to the output
                layers += [nn.ReLU(inplace=True)]
            in_features = out_features
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

# Deterministic Fourier feature encoder
class FFEncoder:
    def __init__(self, freqs=[1.], include_linear=False):
        self.freqs = freqs
        self.dim = int(include_linear) + len(freqs) * 2
        self.include_linear = include_linear
        print(f'Input dim = {self.dim}')
    
    def encode_single(self, x):
        features = [torch.tensor([x])] if self.include_linear else []
        for freq in self.freqs:
            c = 2. * math.pi * freq
            features += [torch.tensor([torch.sin(c * x), torch.cos(c * x)])]

        return torch.cat(features)
    
    def encode_batch(self, x_batch):
        return torch.cat([
            self.encode_single(x) for x in x_batch
        ]).view(len(x_batch), self.dim)

# Target sin function
def create_training_data(
    num=100,
    freq=1.
):
    inputs = torch.linspace(0.0, 1.0, num, dtype=torch.float32)
    targets = torch.sin(2.*math.pi*freq * inputs)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    return inputs, targets, loader

def visualize(
    input_freq,
    target_freq,
    inps, trgs, enc_inps, outs,
    dirpath='figures',
    filename='figure.png'
):
    plt.clf()
    plt.title(f'Input FF freq = {input_freq} and target freq = {target_freq}')
    plt.plot(inps, trgs, label='Target', color='r')
    plt.plot(inps, enc_inps[:,0], label='Input', color='orange')
    plt.plot(inps, outs, label='Network', linestyle='--', color='b')
    plt.legend()
    os.makedirs(dirpath)
    plt.savefig(os.path.join(dirpath, filename))
    # plt.show()

def run_experiment(
    experiment_freq_list,
    target_freq,
    device,
    num_epochs=100,
    linear_net=True,
):
    for expr_idx, input_freqs in enumerate(experiment_freq_list):
        enc = FFEncoder(freqs=input_freqs)

        if linear_net:
            net = Net(in_dim=enc.dim, num_hidden=0).to(device)
        else:
            net = Net(in_dim=enc.dim, num_hidden=3, hidden_width=128, apply_activation=True).to(device)
        
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

        inps, trgs, loader = create_training_data(num=500, freq=target_freq)
        enc_inps = enc.encode_batch(inps)

        for epoch_idx in range(1, num_epochs+1):
            net.train()
            for sample_batched in loader:
                input = enc.encode_batch(sample_batched[0]).to(device)
                target = sample_batched[1].to(device)
                optimizer.zero_grad()
                output = torch.flatten(net(input))
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            if epoch_idx % 10 == 0:
                net.eval()
                loss = 0.
                for input, target in zip(enc_inps, trgs):
                    input = input.to(device)
                    target = torch.tensor([target]).to(device)
                    output = net(input)
                    loss += criterion(output, target)
                loss /= len(inps)
                print(f'Epoch no.{epoch_idx}: average loss = {loss}')

        net.eval()
        outs = net(enc_inps.to(device)).cpu().detach()

        net_type_str = 'linear' if linear_net else 'nonlinear'
        visualize(input_freqs[0], target_freq, inps, trgs, enc_inps, outs,
            filename=f'figure-{net_type_str}-{expr_idx}.png')


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device='cuda')
        print(f'Using {device_name}')
    else:
        device = torch.device('cpu')
        print('Using cpu')

    run_experiment(
        experiment_freq_list=[[0.8], [1.1], [1.25], [2.0], [2.5]],
        target_freq=2.5,
        device=device,
        num_epochs=500,
        linear_net=True
    )
    run_experiment(
        experiment_freq_list=[[0.8], [1.1], [1.25], [2.0], [2.5]],
        target_freq=2.5,
        device=device,
        num_epochs=500,
        linear_net=False
    )

if __name__ == '__main__':
    main()