import os, math
import extract_bvh_array
import numpy
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layer):
        super().__init__()
        hidden_size = input_size * 3
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        #layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ELU(inplace=True))
        for _ in range(num_hidden_layer):
            layers.append(nn.Linear(hidden_size, hidden_size))
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ELU(inplace=True))
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weighted_mse_loss(input, target, weight):
    return torch.sum((weight * (input - target)) ** 2)

def main():
    device = 'cpu'
    #path = os.path.join( os.getcwd(), "asset", "03_04_walk_on_uneven_terrain.bvh")
    #path = os.path.join( os.getcwd(), "asset", "walk.bvh")
    #path = os.path.join( os.getcwd(), "asset", "03_02_walk_on_uneven_terrain.bvh")
    #path = os.path.join( os.getcwd(), "asset", "03_01_walk_on_uneven_terrain.bvh")
    path = os.path.join( os.getcwd(), "asset", "06_08.bvh")
    np_trg = extract_bvh_array.extract_bvh_array(path)
    np_trg = np_trg[1:,]
    pt_trg = torch.from_numpy(np_trg).float()
    print(pt_trg.shape)

    np_weight = numpy.ones([np_trg.shape[1]])
    np_weight[:3] = 10
    pt_weight = torch.from_numpy(np_weight)
    print(pt_weight.data)
    pt_weight = pt_weight.to(device)
    print(pt_weight.shape)

    cycles = []
    for itr in range(7):
        net = MLP(1+len(cycles)*2,np_trg.shape[1],num_hidden_layer=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        tmp_list = []
        tmp_list.append( torch.linspace(0.0, 1.0, pt_trg.shape[0], dtype=torch.float32) )
        for cycle in cycles:
            tmp_list.append( torch.sin(2.*math.pi*cycle*tmp_list[0]) )
            tmp_list.append( torch.cos(2.*math.pi*cycle*tmp_list[0]) )
        pt_in = torch.stack(tmp_list,dim=1) # pt_in.reshape([*pt_in.shape,1])
        print(pt_in.shape)
        loader = DataLoader(TensorDataset(pt_in, pt_trg), batch_size=20, shuffle=True)

        for iepoch in range((itr+1)*1000):
            net.train()
            for sample_batched in loader:
                batch_in = sample_batched[0].to(device)
                batch_trg = sample_batched[1].to(device)
                optimizer.zero_grad()
                batch_out = net.forward(batch_in)
                loss = weighted_mse_loss(batch_out, batch_trg, weight=pt_weight)
                loss.backward()
                optimizer.step()
            net.eval()
            if iepoch % 100 == 0:
                with torch.no_grad():
                    pt_out = net.forward(pt_in)
                    loss = weighted_mse_loss(pt_out, pt_trg, weight=pt_weight)
                    print("   ",iepoch, loss.data.item())

        with torch.no_grad():
            pt_out = net.forward(pt_in)
            np_out = pt_out.numpy()
            print(np_out.shape)

        path_save,_ = os.path.splitext(path)
        path_save = path_save + "_{}.bvh".format(itr)
        extract_bvh_array.swap_data_in_bvh(path_save, np_out, path)

        np_diff = np_out - np_trg
        # L0 norm
        print("cycles",cycles)
        print("compression ratio: ",np_trg.size/count_parameters(net))
        print("trans_diff:",np_diff[:,:3].max())
        print("angle_diff:",np_diff[:,3:].max())

        # pca and dct
        deviation = np_weight * (np_diff - np_diff.mean(axis=0))
        eig_val, eig_vec = numpy.linalg.eig(deviation.transpose() @ deviation)
        eig_vec = eig_vec.astype(numpy.float64)
        history0 = deviation @ eig_vec[:,0]
        #plt.plot(history0)
        #plt.show()
        #print(numpy.abs(dct(history0)))
        new_cycle = numpy.abs(dct(history0)).argmax() * 0.5
        cycles.append(new_cycle)



if __name__ == "__main__":
    main()