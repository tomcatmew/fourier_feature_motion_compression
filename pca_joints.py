import os, sys
import numpy
import matplotlib.pyplot as plt
import extract_bvh_array

if __name__ == "__main__":
    path = os.path.join( os.getcwd(), "asset", "walk.bvh")
    data = extract_bvh_array.extract_bvh_array(path)
    joint_data = data[:,3:]
    print(data.shape, joint_data.shape)
    mean_joint_data = joint_data - joint_data.mean(axis=0)
    eig_val, eig_vec = numpy.linalg.eig(mean_joint_data.transpose() @ mean_joint_data)
    eig_vec0 = eig_vec[:,0]
    history0 = mean_joint_data @ eig_vec0
    fk = numpy.fft.fft(history0)
    freq = numpy.fft.fftfreq(history0.shape[0])
    plt.plot(history0)
    #plt.plot(freq,numpy.abs(fk))
    #plt.xlim(0,0.5)
    plt.show()