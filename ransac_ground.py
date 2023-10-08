import random
import struct
import open3d as o3d
import numpy as np
from math import acos, sin, cos, fabs, sqrt, log
from matplotlib import pyplot as plt
from pandas import DataFrame
from pyntcloud import PyntCloud
from mpl_toolkits.mplot3d import Axes3D

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
def points2open3d(points):
    origin_points_df = DataFrame(points,columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    return point_cloud_o3d


def getData(filepath, row_need=1000):
    """
    加载数据并且取其中的部分行
    """
    data = np.loadtxt(filepath, delimiter=",")
    row_total = data.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)

    return data[row_sequence[0:row_need], :]


def solve_plane(A, B, C):
    """
    求解平面方程
    :param A: 点A
    :param B: 点B
    :param C: 点C
    :return: Point(平面上一点),Quaternion(平面四元数),Nx(平面的法向量)
    """

    # 两个常量
    N = np.array([0, 0, 1])
    Pi = np.pi

    # 计算平面的单位法向量，即BC 与 BA的叉积
    Nx = np.cross(B - C, B - A)
    Nx = Nx / np.linalg.norm(Nx)

    # 计算单位旋转向量与旋转角（范围为0到Pi）
    Nv = np.cross(Nx, N)
    angle = acos(np.dot(Nx, N))

    # 考虑到两个向量夹角不大于Pi/2，这里需要处理一下
    if angle > Pi / 2.0:
        angle = Pi - angle
        Nv = -Nv

    # FIXME: 此处如何确定平面上的一个点？？？
    # Point = (A + B + C) / 3.0
    Point = B
    # 计算单位四元数
    Quaternion = np.append(Nv * sin(angle / 2), cos(angle / 2))

    # print("旋转向量:\t", Nv)
    # print("旋转角度:\t", angle)
    # print("对应四元数:\t", Quaternion)

    return Point, Quaternion, Nx


def solve_distance(M, P, N):
    """
    求解点M到平面(P,Q)的距离
    :param M: 点M
    :param P: 平面上一点
    :param N: 平面的法向量
    :return: 点到平面的距离
    """

    # 从四元数到法向量
    # A = 2 * Q[0] * Q[2] + 2 * Q[1] * Q[3]
    # B = 2 * Q[1] * Q[2] - 2 * Q[0] * Q[3]
    # C = -Q[0] ** 2 - Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2
    # D = -A * P[0] - B * P[1] - C * P[2]

    # 为了计算简便，直接使用求解出的法向量
    A = N[0]
    B = N[1]
    C = N[2]
    D = -A * P[0] - B * P[1] - C * P[2]

    return fabs(A * M[0] + B * M[1] + C * M[2] + D) / sqrt(A ** 2 + B ** 2 + C ** 2)


def RANSAC(data):
    """
    使用RANSAC算法估算模型
    """
    # 数据规模
    SIZE = data.shape[0]
    # 迭代最大次数，每次得到更好的估计会优化iters的数值，默认10000
    iters = 1000
    # 数据和模型之间可接受的差值，默认0.25
    sigma = 0.5
    # 内点数目
    pretotal = 0
    # 希望的得到正确模型的概率，默认0.99
    Per = 0.999
    # 初始化一下
    P = np.array([])
    Q = np.array([])
    N = np.array([])
    for i in range(iters):
        # 随机在数据中选出三个点去求解模型
        sample_index = random.sample(range(SIZE), 3)
        P, Q, N = solve_plane(data[sample_index[0]], data[sample_index[1]], data[sample_index[2]])

        # 算出内点数目
        total_inlier = 0
        for index in range(SIZE):
            distance=solve_distance(data[index], P, N)
            idx_ground = (distance <= sigma)
            if distance< sigma:
                total_inlier = total_inlier + 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            iters = log(1 - Per) / log(1 - pow(total_inlier / SIZE, 2))
            pretotal = total_inlier

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > SIZE / 2:
            break
    idx_segmented = np.logical_not(idx_ground)

    # return P, Q, N
    return data[idx_ground], data[idx_segmented]


def draw(ground_cloud,segmented_cloud):
    point_cloud_o3d_ground = points2open3d(ground_cloud)
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])


    #显示segmentd_points示地面点云
    point_cloud_o3d_segmented = points2open3d(segmented_cloud)
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])

    o3d.visualization.draw_geometries([point_cloud_o3d_ground,point_cloud_o3d_segmented])
    o3d.visualization.draw_geometries([point_cloud_o3d_segmented])
    o3d.visualization.draw_geometries([point_cloud_o3d_ground])


# def test():
#     A = np.random.randn(3)
#     B = np.random.randn(3)
#     C = np.random.randn(3)
#
#     P, Q, N = solve_plane(A, B, C)
#     print("Plane:\t", P, Q)
#
#     D = np.random.randn(3)
#     print("Point:\t", D)
#
#     d = solve_distance(D, P, N)
#     print("Distance:\t", d)


if __name__ == '__main__':
    filename = '/media/bopang/PBDATA/dataset/kitti/odometry/data/sequences/00/velodyne/000000.bin'  # 数据集路径
    print('clustering pointcloud file:', filename)
    data = read_velodyne_bin(filename)  # 读取数据点
    ground_cloud,segmented_cloud = RANSAC(data)
    draw(ground_cloud,segmented_cloud)