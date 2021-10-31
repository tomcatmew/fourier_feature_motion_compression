import glob
import os
import sys
import numpy
import torch
sys.path.append(os.path.join(
    os.path.dirname(__file__), 'external', 'delfem2-python-bindings'))
print(__file__)
print(sys.path)
import delfem2
from delfem2.delfem2 import BVH
from delfem2.delfem2 import get_parameter_history_bvh
from delfem2.delfem2 import get_joint_position_history_bvh
from delfem2.delfem2 import set_parameter_history_bvh_double
import bvh_weights
import compress

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compress_cmu(path_dir):
    bvh_paths = glob.glob(path_dir + '/*/*/*.bvh')
    # bvh_paths = sorted(bvh_paths)
    print("path_dir: ",path_dir)
    print("num_bvh: ",len(bvh_paths))
    print("bvh_paths: ",bvh_paths)
    with open('log.csv', 'w'):
        pass
    for bvh_path in bvh_paths:
        print("#######################")
        bvh = BVH(bvh_path)
        scale, np_weights = bvh_weights.bvh_weights(bvh)
        print(bvh_path, scale)
        np_trg = get_parameter_history_bvh(bvh)[1:,:]  # skip the first frame
        print(np_trg.shape, np_trg.dtype)
        frame_jp0 = get_joint_position_history_bvh(bvh)[1:,:,:] # skip the first frame

        cycles = []
        net = None
        for itr in range(12):
            net_new = compress.MLP(1+len(cycles)*2,np_trg.shape[1],num_hidden_layer=1)
            if net is not None:
                compress.copy_net_weights(net_new, net)
            net = net_new
            np_out = compress.compress(net, cycles, np_trg,np_weights)
            cycles.append( compress.new_cycle(np_out-np_trg, np_weights) )
            # convergene
            set_parameter_history_bvh_double(bvh, np_out.astype(numpy.float64))
            frame_jp1 = get_joint_position_history_bvh(bvh)
            jnt_diff_ratio = (frame_jp0-frame_jp1).max() / scale
            cmp_ratio = np_trg.size / count_parameters(net)
            print(itr, cmp_ratio, jnt_diff_ratio)
            if jnt_diff_ratio < 0.01:
                break

        with open('log.csv', 'a') as f:
            f.write(os.path.basename(bvh_path)+","+str(cmp_ratio)+","+str(jnt_diff_ratio)+"\n")

        '''
        assert len(apps) == len(nets)
        for inet in range(len(nets)):
            app = apps[inet]
            np_diff = app - np_trg
            set_parameter_history_bvh_double(bvh, app.astype(numpy.float64))
            frame_jp1 = get_joint_position_history_bvh(bvh)
            cmp_ratio = np_trg.size / count_parameters(nets[inet])
            jnt_diff_ratio = (frame_jp0-frame_jp1).max() / scale
            print("##")
            print("  compression ratio: ",cmp_ratio)
            print("  trans_diff:     ",np_diff[:,:3].max())
            print("  angle_diff:     ",np_diff[:,3:].max())
            # print("  joint diff l2   ", numpy.linalg.norm(frame_jp0-frame_jp1))
            print("  joint diff l0   ", jnt_diff_ratio)
            with open('log.csv', 'a') as f:
                f.write(os.path.basename(bvh_path)+","+str(cmp_ratio)+","+str(jnt_diff_ratio)+"\n")
        '''

def test0(path_dir):
    bvh_paths = glob.glob(path_dir + '/*/*/*.bvh')
    bvh_paths = sorted(bvh_paths)
    bvh0 = BVH()
    bvh0.open(bvh_paths[0])
    frame_jp0 = get_joint_position_history_bvh(bvh0)

    bvh1 = BVH()
    bvh1.open(bvh_paths[0])
    frame_jp1 = get_joint_position_history_bvh(bvh1)


    # print(frame_jp0)
    # print((frame_jp0-frame_jp1).max())


if __name__ == "__main__":
    path_dir = '/Volumes/CmuMoCap'
    # test0(path_dir)
    # path_dir = '/media/nobuyuki/CmuMoCap'
    compress_cmu(path_dir)
