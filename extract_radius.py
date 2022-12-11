import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

def select_points(tangent, src, tgt, threshold = 0.5):
    dis = (src.reshape(1, 3) - tgt) * tangent.reshape(1, 3)
    mask = abs(dis.sum(-1)) < threshold
    return mask

def fitting_radius(center, points, threshold = 0.3):
    r_s = (((points - center.reshape(1,3))**2).sum(-1)**0.5).reshape(-1, 1)
    vote = r_s - r_s.T
    vote = abs(vote) < threshold

    r = r_s[vote.sum(-1).argmax()]
    mask = vote[vote.sum(-1).argmax()]
    return r, mask

def make_circle(center, r, normal, threshold = 0.05):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius= r, resolution = 100)
    sphere = sphere.sample_points_poisson_disk(1000)
    points = np.array(sphere.points)
    mask = abs(((points/r) * normal.reshape(-1,3)).sum(-1)) < threshold
    points = points[mask]
    points += center
    return points




if __name__ == "__main__":
    pcd0 = o3d.io.read_point_cloud('01t.ply')
    pcd1 = o3d.io.read_triangle_mesh('label.ply')
    pcd1 = pcd1.sample_points_poisson_disk(2000)


    normal = np.gradient(np.array(pcd0.points))[0]
    tangent = normal / ((normal**2).sum(-1)**0.5).reshape(-1, 1)
    pcd0.normals = o3d.utility.Vector3dVector(tangent)

    point0 = np.array(pcd0.points)
    point1 = np.array(pcd1.points)

    pcd0.paint_uniform_color([1, 0, 0])
    pcd1.paint_uniform_color([0, 1, 0])

    Radius = []

    #extract putative points
    for i in range(2, len(point0)-2):
        mask1 = select_points(tangent[i], point0[i], point1)
        r, mask2 = fitting_radius(point0[i], point1[mask1])
        Radius.append(r)

        # point_circle = o3d.geometry.PointCloud()
        # point_circle.points = o3d.utility.Vector3dVector(point1[mask1] + 0.01)
        # point_circle.paint_uniform_color([0,0,1])
        # o3d.visualization.draw_geometries([pcd0, pcd1, point_circle])

        point_circle = o3d.geometry.PointCloud()
        point_circle.points = o3d.utility.Vector3dVector(make_circle(point0[i], r, tangent[i]))
        point_circle.paint_uniform_color([0,0,1])
        o3d.visualization.draw_geometries([pcd0, pcd1, point_circle])


    o3d.visualization.draw_geometries([pcd0, pcd1])