import os, sys
import numpy
import matplotlib.pyplot as plt
import extract_bvh_array

if __name__ == "__main__":
    path = os.path.join( os.getcwd(), "asset", "walk.bvh")
    data = extract_bvh_array.extract_bvh_array(path)
    joint_data = data[1:,3:]
    print(data.shape, joint_data.shape,joint_data.size)

    mean_joint_data = joint_data - joint_data.mean(axis=0)
    eig_val, eig_vec = numpy.linalg.eig(mean_joint_data.transpose() @ mean_joint_data)
    eig_vec0 = eig_vec[:,0]
    history0 = mean_joint_data @ eig_vec0
    fk = numpy.fft.fft(history0)
    freq = numpy.fft.fftfreq(history0.shape[0])
    plt.plot(history0)
    plt.show()

    # plt.plot(freq,fk)
    # plt.xlim(-0.1,+0.1)
    plt.plot(fk)
    plt.show()

    U, s, V = numpy.linalg.svd(joint_data, full_matrices=True)
    print(U.shape,s.shape,V.shape)
    print(s)
    joint_data1 = U[:,:s.shape[0]].dot(numpy.diag(s)).dot(V)
    assert (joint_data - joint_data1).max() < 1.0e-3

    comp_ratios = []
    errors = []
    for num_basis in range(1,joint_data.shape[1]):
        joint_data1 = U[:,:num_basis].dot(numpy.diag(s[:num_basis])).dot(V[:num_basis,:])
        num_param = num_basis * (joint_data.shape[0] + joint_data.shape[1] + 1)
        comp_ratios.append( joint_data.size / num_param )
        errors.append((joint_data - joint_data1).max())

    fig,ax = plt.subplots()
    ax.plot(comp_ratios,errors)
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("Euler angle error in degree")
    plt.show()



