import os, sys
import numpy
import matplotlib.pyplot as plt
import extract_bvh_array
import fourier_feature_network
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    test_parameter = [2, 4, 8, 16, 32, 64, 128, 256]
    plt_labels = []
    for i in test_parameter:
        plt_labels.append("B = " + str(i))

    path = os.path.join(os.getcwd(), "asset", "126_14_stretch.bvh")
    data = extract_bvh_array.extract_bvh_array(path)
    data_q = numpy.zeros((data.shape[0], int(data.shape[1] / 3) * 4 - 1))
    print(data.shape)
    print(data_q.shape)

    line_count = 0
    for i in data:
        # root position doesn't need to convert
        pos = i[0:3]
        data_q[line_count][0:3] = pos
        for j in range(1, int(len(i) / 3)):
            # convert euler angles to quaternion
            euler = i[j * 3:j * 3 + 3]
            rot = Rotation.from_euler('zyx', euler, degrees=True)
            rot_quat = rot.as_quat()
            data_q[line_count][3 + (j - 1) * 4: 3 + ((j - 1) * 4) + 4] = rot_quat
        line_count += 1

    # phase generator
    phase = numpy.linspace(0, 1, data.shape[0], endpoint=False)
    phase_double = numpy.repeat(phase[:, numpy.newaxis], 2, axis=1)

    for hyper_para_b in range(len(test_parameter)):
        train_data = [phase_double, data_q]

        x, y = torch.tensor(train_data[0]).reshape(-1, 2), torch.tensor(train_data[1]).reshape(-1, 127)
        x, y = x.float().cuda(), y.float().cuda()

        model = fourier_feature_network.MLP(mapping_size=test_parameter[hyper_para_b] * 2).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss = nn.MSELoss()

        B = torch.randn(2, test_parameter[hyper_para_b]).cuda() * 10

        input_x = fourier_feature_network.fourier_map(x, B)

        running_loss_list = []
        running_loss = 0.0
        running_correct = 0.0
        for i in tqdm(range(300)):
            ypred = model(input_x)
            l = loss(ypred, y)
            opt.zero_grad()
            l.backward()
            opt.step()

            running_loss = l.item()
            running_loss_list.append(running_loss)
            running_correct += (ypred == y).sum().item()

        model.cuda().eval()
        with torch.no_grad():
            y_predict = model(input_x)

        evaluate_y = y_predict.data.cpu().numpy()
        # numpy.savetxt("output_bvh", evaluate_y, fmt='%.6f')

        plt.plot(running_loss_list, color=color_sequence[hyper_para_b])

    plt.legend(plt_labels)
    plt.xlabel("Training Iterations")
    plt.ylabel("MSELoss")
    plt.show()
