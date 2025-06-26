"""
The idea of this code is to illustrate tension in molecules. This is done by using the hessian (Force constant matrix) as measure. For plotting a fake nmd file is created for VMD. By setting the mode displacements to this measure, VMDs and Normal Mode Wizards beta coloring method can be used for illustration. Do not plot the eigenmodes. They are meaningless.
"""
import sys
import tmoutproc as top
import numpy as np
import eigenchannel_utils as eu

def calculate_tension(hessian):
    """
    Calculates tension from hessian by adding the absolute values of the force acting on each degree of freedom. This is done to consider non-restoring force components.
    Args:
        hessian:

    Returns:
        tension (np.array): Tension array
    """
    tension = np.zeros((hessian.shape[0],1))
    for i in range(0,hessian.shape[0]):
        tension[i] = np.sum(np.abs(hessian[i,:]))-hessian[i,i]

    return tension



if __name__ == '__main__':
    hessian_path = sys.argv[1]
    coord_path = sys.argv[2]
    nmd_output_path = sys.argv[3]

    coord = top.read_coord_file(coord_path)
    hessian = top.read_hessian(hessian_path, len(coord))

    coord = eu.filter_coord(coord)
    min_index = 16*3
    max_index = hessian.shape[0]-16*3
    hessian = hessian[min_index:max_index, min_index:max_index]



    tension = calculate_tension(hessian)

    eu.write_nmd_file(nmd_output_path, coord, tension, 1, use_mass_scaling=True)

