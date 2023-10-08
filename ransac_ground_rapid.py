
import numpy as np
import open3d as o3d
import struct
from pandas import DataFrame
from pyntcloud import PyntCloud
import math
import random
import time
from tqdm import tqdm

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def fit_plane(data):
    sample_indices = random.sample(range(len(data)), 3)
    points = data[sample_indices]
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    d = -np.dot(normal, points[0])
    return normal, d

def distance_to_plane(points, normal, d):
    distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
    return distances

# P #希望的到正确模型的概率
# n = len(data)    #点的数目
# iters    #最大迭代次数  000002.bin：10
# sigma    #数据和模型之间可接受的最大差值
def ground_segmentation(data, max_iterations=1000, sigma=0.1, P=0.999, outline_ratio=0.5):
    n = len(data)
    best_model = None
    best_inliers = 0

    for _ in range(max_iterations):
        plane_normal, plane_d = fit_plane(data)
        distances = distance_to_plane(data, plane_normal, plane_d)
        inliers = distances <= sigma

        if np.sum(inliers) > best_inliers:
            best_inliers = np.sum(inliers)
            best_model = (plane_normal, plane_d)
            max_iterations = np.log(1 - P) / np.log(1 - pow(best_inliers / n, 3))

        if best_inliers > n * (1 - outline_ratio):
            break

    idx_ground = distance_to_plane(data, *best_model) <= sigma
    idx_segmented = ~idx_ground

    return data[idx_ground], data[idx_segmented]

def points2open3d(bin_points):
    origin_points_df = DataFrame(bin_points,columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    return point_cloud_o3d



def ransac_show_demo():
    iteration_num = 1    #文件数

    # for i in range(iteration_num):
    filename = '/media/bopang/PBDATA/dataset/kitti/odometry/data/sequences/00/velodyne/000000.bin'         #数据集路径
    print('clustering pointcloud file:', filename)

    origin_points = read_velodyne_bin(filename)   #读取数据点
    # o3d.visualization.draw_geometries([bin2open3d(origin_points)]) # 显示原始点云



    # 地面分割


    start_time = time.time()  # 记录起始时间

    # 执行需要计时的代码
    # for _ in tqdm(range(10000), desc="Processing", ncols=100):
    #     ground_points, segmented_points = ground_segmentation(data=origin_points)
    ground_points, segmented_points = ground_segmentation(data=origin_points)
    end_time = time.time()  # 记录结束时间

    point_cloud_o3d_ground = points2open3d(ground_points)
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])


    #显示segmentd_points示地面点云
    point_cloud_o3d_segmented = points2open3d(segmented_points)
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])

    o3d.visualization.draw_geometries([point_cloud_o3d_ground,point_cloud_o3d_segmented])
    o3d.visualization.draw_geometries([point_cloud_o3d_segmented])
    o3d.visualization.draw_geometries([point_cloud_o3d_ground])


    elapsed_time = end_time - start_time  # 计算经过的时间（秒）
    print(f"经过的时间：{elapsed_time}秒")

def main():
    ransac_show_demo()


if __name__ == '__main__':
    main()


