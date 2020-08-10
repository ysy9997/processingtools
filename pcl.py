import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2


def progress_bar(progress: int, length: int, bar_length: int = 50, finish_mark: str = 'progress finished!'):
    """
    print progress
    :param progress: the number of present progress
    :param length: the number of total progress
    :param bar_length: bar length
    :param finish_mark: print string what you want when progress finish
    :return: return: True
    """

    progress = progress + 1
    progress_per = progress / length * 100
    progress_per_str = str(int(progress_per * 10) / 10)
    bar = '█' * int(bar_length / 100 * progress_per)
    space = '░' * (bar_length - int(bar_length / 100 * progress_per))

    print('\r|%s%s|    %s%%    %d/%d' % (bar, space, progress_per_str, progress, length), end='')
    if progress == length: print('\n' + finish_mark)

    return True


def img2xyz(img, depth):
    """

    """
    h, w = img.shape[:2]
    depth = cv2.resize(depth, (w, h))

    x = np.linspace(0, w - 1, w, dtype=float)
    y = np.linspace(0, h - 1, h, dtype=float)
    x, y = np.meshgrid(x, y)
    xy = np.concatenate((y.reshape(h, w, 1), x.reshape(h, w, 1)), axis=2)

    xy[:, :, 0] = xy[:, :, 0].astype(float) / h * np.pi
    xy[:, :, 1] = xy[:, :, 1].astype(float) / w * np.pi * 2
    depth = depth.astype(float) / 255 * 30
    rtp = np.concatenate((np.expand_dims(depth[:, :, 0], axis=-1), xy), axis=2)

    x = rtp[:, :, 0] * np.sin(rtp[:, :, 1]) * np.cos(rtp[:, :, 2])
    y = rtp[:, :, 0] * np.sin(rtp[:, :, 1]) * np.sin(rtp[:, :, 2])
    z = rtp[:, :, 0] * np.cos(rtp[:, :, 1])

    return x, y, z

image = cv2.imread("/media/vcl/Seagate/mmmm/mugung/filtered_l.png")
depth = cv2.imread("/media/vcl/Seagate/mmmm/mugung/depth_comb-WLS-WLS.png")

depth = cv2.medianBlur(depth, 5)
#depth = cv2.GaussianBlur(depth, (5, 5), 0)
#depth = cv2.GaussianBlur(depth, (5, 5), 0)
#image = cv2.resize(image, (2048, 1024))

x, y, z = img2xyz(image, depth)
xyz = []
color = []
h, w, _ = image.shape
for i in range(w):
    for j in range(h):
        xyz.append([x[j, i], y[j, i], z[j, i]])
        color.append([image[j, i, 2] / 255, image[j, i, 1] / 255, image[j, i, 0] / 255])
    progress_bar(i, w)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])

"""
color_raw = o3d.io.read_image("/home/vcl/dataset/360dataset/MPEG/v0_bgr/0.png")
depth_raw = o3d.io.read_image("/home/vcl/dataset/360dataset/drive-download-20200714T112742Z-001/v0_4096_2048_0_8_1000_0_420_10b.yuv.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

#pcd = get_spherical_pcd(pcd)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
"""