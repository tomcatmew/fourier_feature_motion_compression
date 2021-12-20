
import os
import numpy

def extract_bvh_array(path: str) -> numpy.ndarray:
    assert os.path.isfile(path)
    with open(path) as f:
        for s_line in f:
            if s_line.startswith('MOTION'):
                break
        s_line = f.readline()
        nframe = int(s_line.split()[1])
        _ = f.readline()
        arr = []
        for iframe in range(nframe):
            a = [float(s) for s in f.readline().split()]
            arr.append(a)
    return numpy.array(arr)


def swap_data_in_bvh(
        path_save: str,
        data: numpy.ndarray,
        path_org: str):
    assert os.path.isfile(path_org)
    fout = open(path_save,mode='w')
    with open(path_org) as fin:
        for s_line in fin:
            fout.write(s_line)
            if s_line.startswith('MOTION'):
                break
        #
        _ = fin.readline()
        fout.write("Frames: {}\n".format(data.shape[0]))
        #
        s_line = fin.readline()
        fout.write(s_line)
        #
        for iframe in range(data.shape[0]):
            for iparam in range(data.shape[1]):
                fout.write(str(data[iframe][iparam])+" ")
            fout.write("\n")
    fout.close()

if __name__ == "__main__":
    path = os.path.join( os.getcwd(), "asset", "walk.bvh")
    params = extract_bvh_array(path)
    print(params)