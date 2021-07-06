import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

capture_name = 'capture_BlueprintOffice_Taro_none_1625551938'
image_number = '00000003'

depth_image_path = './{}/{}_depthL.png'.format(capture_name, image_number)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

rgb_image_path = './{}/{}_rgbL.png'.format(capture_name, image_number)
rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_UNCHANGED)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

attributes_path = './{}/{}_attributes.json'.format(capture_name, image_number)
with open(attributes_path, 'r') as f:
    attributes = json.load(f)

UE4_to_ImageXYZ = np.asarray([
    [0, 10, 0, 0],
    [0, 0, -10, 0],
    [10, 0, 0, 0],
    [0, 0, 0, 1]
])

ImageXYZ_to_UE4 = np.linalg.inv(UE4_to_ImageXYZ)

camera_intrinsics = np.asarray(attributes['camera_intrinsics'], dtype=np.float32)
camera_intrinsics_inv = np.linalg.inv(camera_intrinsics)

camera_to_world = UE4_to_ImageXYZ.dot(np.asarray(attributes['camera_to_world'], dtype=np.float32).dot(ImageXYZ_to_UE4))
if attributes['left_camera_is_aligned_to_center']:
    left_camera_to_world = camera_to_world
    right_camera_to_world = None  # TODO: impl
else:
    left_camera_to_world = None  # TODO: impl
    right_camera_to_world = None  # TODO: impl

world_to_left_camera = np.linalg.inv(left_camera_to_world)

joints_u = []
joints_v = []

joints_name = [
    #'head_to_world',
    'right_upperarm_to_world',
    'right_lowerarm_to_world',
    'right_hand_to_world',
    'right_middle_01_to_world',
    #'right_middle_02_to_world',
    #'right_middle_03_to_world',
    # TODO: we'd like to get finger tip location, which we need to compute/estimate from middle_03 pose and the distance between middle02 and middle03.
    'left_upperarm_to_world',
    'left_lowerarm_to_world',
    'left_hand_to_world',
    'left_middle_01_to_world',
    #'left_middle_02_to_world',
    #'left_middle_03_to_world',
]

for name in joints_name:
    bone_to_world = UE4_to_ImageXYZ.dot(np.asarray(attributes[name], dtype=np.float32).dot(ImageXYZ_to_UE4))
    bone_to_camera = world_to_left_camera.dot(bone_to_world)
    bone_xyz = bone_to_camera[:3, 3]

    bone_uvd = camera_intrinsics.dot(bone_xyz)
    bone_uvd[:2] /= bone_uvd[2:3]

    joints_u.append(bone_uvd[0])
    joints_v.append(bone_uvd[1])

plt.imshow(depth_image, interpolation='nearest')
plt.plot(joints_u, joints_v, marker='*', linestyle='None')
plt.colorbar()

plt.show()



w = depth_image.shape[1]
h = depth_image.shape[0]

u = np.arange(w)
v = np.arange(h)

uvd = np.stack(np.meshgrid(u, v) + [depth_image,], axis=-1)
uvdrgb = np.concatenate([uvd, rgb_image], axis=-1).reshape(-1, 6)

uvdrgb[:, :2] *= uvdrgb[:, 2:3]
uvdrgb = uvdrgb[(1 < uvdrgb[:, 2]) & (uvdrgb[:, 2] < 65535)]  # remove pixels with invalid depth

xyz = np.einsum('ij,pj->pi', camera_intrinsics_inv, uvdrgb[:, :3])
xyzrgb = np.concatenate([xyz, uvdrgb[:, 3:]], axis=-1)

# you can view ply file with MeshLab for example.
ply_path = './{}/{}_pointcloud.ply'.format(capture_name, image_number)
with open(ply_path, 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(xyz.shape[0]))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')
    for data in xyzrgb.tolist():
        f.write('{} {} {} {:.0f} {:.0f} {:.0f}\n'.format(*data))