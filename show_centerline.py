import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

# import openmesh as om
#
# mesh = om.TriMesh()
# om.read_mesh(mesh, 'label.stl')
# om.write_mesh(mesh, 'label.ply')


if __name__ == "__main__":
    pcd0 = o3d.io.read_point_cloud('01t.ply')
    pcd1 = o3d.io.read_triangle_mesh('label.ply')
    pcd1 = pcd1.sample_points_poisson_disk(2000)


    # pcd0.estimate_normals()

    # x, y, z = np.array(pcd0.points)[:, 0], np.array(pcd0.points)[:, 1], np.array(pcd0.points)[:, 2]
    normal = np.gradient(np.array(pcd0.points))[0]
    normal = normal / ((normal**2).sum(-1)**0.5).reshape(-1, 1)
    pcd0.normals = o3d.utility.Vector3dVector(normal)

    pcd0.paint_uniform_color([1, 0, 0])
    pcd1.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd0, pcd1])