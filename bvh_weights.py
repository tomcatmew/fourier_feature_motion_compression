import math
import os
import sys

import numpy

sys.path.append(os.path.join(
    os.path.dirname(__file__), 'external', 'delfem2-python-bindings'))
from delfem2.delfem2 import BVH


def norm_l2(a, b):
    assert len(a) == len(b)
    sum = 0.
    for i in range(len(a)):
        sum += (a[i] - b[i]) * (a[i] - b[i])
    return math.sqrt(sum)


def bvh_weights(bvh: BVH):
    max_dist = numpy.zeros(len(bvh.bones))
    for bone in bvh.bones:
        ibone_next = bone.parent_bone_idx
        list_parent_bone_idx = []
        while ibone_next != -1:
            list_parent_bone_idx.append(ibone_next)
            ibone_next = bvh.bones[ibone_next].parent_bone_idx
        # print(bone.name, bone.position(), list_parent_bone_idx)
        for ip in list_parent_bone_idx:
            dist = norm_l2(bone.position(), bvh.bones[ip].position())
            max_dist[ip] = max(dist, max_dist[ip])

    max_dist *= 2 * math.pi / 360.0
    # print(max_dist)

    weights = numpy.zeros((len(bvh.channels)))
    for ich in range(len(bvh.channels)):
        ch = bvh.channels[ich]
        if ch.is_rot:
            weights[ich] = max_dist[ch.ibone]
        else:
            weights[ich] = 1.
    # print(weights)

    bb = bvh.minmax_xyz()
    # print(bb)
    scale = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
    return scale, weights


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "asset", "walk.bvh")
    bvh = BVH()
    bvh.open(path)
    # bvh.clear_pose()
    scale, weights = bvh_weights(bvh)
    print("skeleton_size", scale, weights)
