
import os
import numpy

def extract_bvh_array(path: str) -> numpy.ndarray:
    assert os.path.isfile(path)
    with open(path) as f:
        is_data = False
        for s_line in f:
            if s_line.startswith('MOTION'):
                break
        s_line = f.readline()
        nframe = int(s_line.split()[1])
        s_line = f.readline()
        arr = []
        for iframe in range(nframe):
            a = [float(s) for s in f.readline().split()]
            arr.append(a)
    return numpy.array(arr)

# remove the first frame of bvh motion array
def extract_bvh_array_v2(path: str) -> numpy.ndarray:
    assert os.path.isfile(path)
    with open(path) as f:
        is_data = False
        for s_line in f:
            if s_line.startswith('MOTION'):
                break
        s_line = f.readline()
        nframe = int(s_line.split()[1]) - 1
        s_line = f.readline()
        s_line = f.readline()
        arr = []
        for iframe in range(nframe):
            a = [float(s) for s in f.readline().split()]
            arr.append(a)
    return numpy.array(arr)

# remove the first frame of bvh motion array and remove the translation of root
def extract_bvh_array_v3(path: str) -> numpy.ndarray:
    assert os.path.isfile(path)
    with open(path) as f:
        is_data = False
        for s_line in f:
            if s_line.startswith('MOTION'):
                break
        s_line = f.readline()
        nframe = int(s_line.split()[1]) - 1
        s_line = f.readline()
        s_line = f.readline()
        arr = []
        for iframe in range(nframe):
            a = [float(s) for s in f.readline().split()[3:]]
            arr.append(a)
    return numpy.array(arr)


if __name__ == "__main__":
    path = os.path.join( os.getcwd(), "asset", "walk.bvh")
    params = extract_bvh_array(path)
    print(params)