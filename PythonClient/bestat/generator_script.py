import os
import time
import numpy as np
import airsim

class CameraInfo():
    def __init__(self, HFOV, baseline):
        self.HFOV = HFOV
        self.baseline = baseline

class LevelInfo():
    def __init__(self,
        position_range,
        auto_exposure_bias_range,
    ):
        self.position_range = position_range
        self.auto_exposure_bias_range = auto_exposure_bias_range

def generate_random_position(range):
    d = np.random.rand(3)
    x = range[0] + (range[1] - range[0]) * d[0]
    y = range[2] + (range[3] - range[2]) * d[1]
    z = range[4] + (range[5] - range[4]) * d[2]
    return airsim.Vector3r(x, y, z)

def generate_random_rotation(axis, range):
    a = angle_to_rad(range[0] + (range[1] - range[0]) * np.random.rand())
    c = np.cos(0.5 * a)
    s = np.sin(0.5 * a)
    return airsim.Quaternionr(axis[0] * s, axis[1] * s, axis[2] * s, c)

def generate_random_quaternion():
    norm = 0.
    while norm < 1e-4:
        d = np.random.randn(3)
        norm = np.linalg.norm(d)
    theta = 2. * np.pi * np.random.rand()
    s = np.sin(theta)
    c = np.cos(theta)
    d *= s / norm
    return airsim.Quaternionr(d[0], d[1], d[2], c)

def angle_to_rad(angle):
    return angle * np.pi / 180.

def generate_random_scalar(range):
    return range[0] + (range[1] - range[0]) * np.random.rand()

def main(
    camera_name, align_to_left_camera,
    image_width, image_heightm, requests,
    level_name, num_capture):

    timestamp = int(time.time())

    client = airsim.VehicleClient()

    camera_info = camera_info_list[camera_name]
    level_info = level_info_list[level_name]

    # camera config
    # TODO: we might add random lense distortion for getting more realistic images.
    lc, rc = (0.0, 1.0) if align_to_left_camera else (-0.5, 0.5)
    client.simSetCameraPose('front_left', airsim.Pose(airsim.Vector3r(0., lc * camera_info.baseline, 0.), airsim.Quaternionr()).UE4ToAirSim())
    client.simSetCameraFov('front_left', camera_info.HFOV)
    client.simSetCameraImageSize('front_left', image_width, image_height)
    client.simSetCameraPose('front_right', airsim.Pose(airsim.Vector3r(0., rc * camera_info.baseline, 0.), airsim.Quaternionr()).UE4ToAirSim())
    client.simSetCameraFov('front_right', camera_info.HFOV)
    client.simSetCameraImageSize('front_right', image_width, image_height)

    # load level
    #client.simLoadLevel(level_name)

    # swapn humans/items
    human_name = 'human'
    human_BP_name = metahumans_bp_path_template.format('Taro')
    item_name = 'item'
    #item_static_mesh_name = 'lunchbox2'
    item_static_mesh_name = 'Banana_LOD0_vfendgyiw'
    #item_static_mesh_name = 'Orange_LOD0_tibhfezva'
    #item_static_mesh_name = 'Red_Apple_LOD0_tgzoahbpa'

    client.simSpawnObject(human_name, human_BP_name, physics_enabled=False, from_actorBP=True)
    client.simSpawnObject(item_name, item_static_mesh_name, physics_enabled=False, from_actorBP=False)

    #time.sleep(3)    # wait few seconds so that objects are completely spawned

    client.simSetSegmentationObjectID(human_name, 1);
    client.simSetSegmentationObjectID(item_name, 2);

    # generate data
    capture_folder = './capture_{}/'.format(timestamp)
    os.makedirs(capture_folder, exist_ok=True)
    for idx in range(num_capture):
        #time.sleep(1)

        # TODO: some metahumans's face are unnaturally lightened in Room environment. check it!

        # configure objects in the scene.
        # we follow the UE4's coordinates/units convention for ease of working with the UE4 environment.
        # airsim uses NED coordinates with unit 'm' while UE4 uses NEU coordinates with unit 'cm'.

        # base position/horizontal rotation where we arragnge the stereo camera, humans, items.
        # here we set the base position to near the bottom center of the booth shelf.
        base_position = generate_random_position(level_info.position_range)
        base_rotation = generate_random_rotation((0, 0, 1), (0, 360))
        #base_rotation = airsim.Quaternionr()
        #base_rotation = generate_random_rotation((0, 0, 1), (90, 90))
        world_to_base = airsim.Pose(base_position, base_rotation)

        # derive the stereo camera position
        camera_roll_error = 0#5
        camera_pitch_error = 0#3
        camera_height_error = 0#5
        camera_horiz_error = 0#3
        #camera_position = generate_random_position((80, 80, 40, 40, 160, 160))
        #camera_rotation = generate_random_rotation((0, 0, 1), (-150, -150))
        camera_position = generate_random_position((-20 - camera_horiz_error, -20 + camera_horiz_error, 0 - camera_horiz_error, 0 + camera_horiz_error, 212.5 - camera_height_error, 212.5 + camera_height_error))
        camera_rotation = generate_random_rotation((0, 0, 1), (-camera_roll_error, camera_roll_error)) * generate_random_rotation((0, 1, 0), (90 - camera_pitch_error, 90 + camera_pitch_error))
        base_to_camera = airsim.Pose(camera_position, camera_rotation)
        world_to_camera = base_to_camera * world_to_base

        # derive the human position
        human_height_diff = 0
        crouching_diff = 0
        human_position = generate_random_position((-40, -40, 0, 0, -human_height_diff - crouching_diff, human_height_diff - crouching_diff))
        #human_position = generate_random_position((-200, -30, -10, 10, -human_height_diff - crouching_diff, human_height_diff - crouching_diff))
        human_rotation_diff = 0#60
        human_rotation = generate_random_rotation((0, 0, 1), (-90 - human_rotation_diff, -90 + human_rotation_diff))
        base_to_human = airsim.Pose(human_position, human_rotation)
        world_to_human = base_to_human * world_to_base

        # derive the human pose
        left_hand_IKposition = airsim.Vector3r(1000, 10000, 500)
        left_hand_rotation = generate_random_position(metahumans_hand_rotation_range)

        # TODO: find natural range
        right_hand_IKposition = airsim.Vector3r(0, 0, -1000)
        right_hand_IKposition = airsim.Vector3r(0, -1000, 0)
        right_hand_IKposition = airsim.Vector3r(-1000, -10000, -500)
        right_hand_rotation = generate_random_position(metahumans_hand_rotation_range)#generate_random_position((-120, -120, 0, 0, 0, 0))

        # TODO: find a way to make crouching pose, natural foots, waist pose

        # TODO: find a way to put item around hand
        item_position = airsim.Vector3r(22, 5, 135)
        base_to_item = airsim.Pose(item_position, generate_random_rotation((0, 0, 1), (-4, -4)))
        world_to_item = base_to_item * world_to_base

        # request pose changes.
        client.simSetVehiclePose(world_to_camera.UE4ToAirSim())
        client.simSetObjectPose(human_name, world_to_human.UE4ToAirSim())
        client.simSetMetahumanPose(human_name, left_hand_IKposition, left_hand_rotation, right_hand_IKposition, right_hand_rotation)
        client.simSetObjectPose(item_name, world_to_item.UE4ToAirSim())

        time.sleep(0.001)  # sometimes images will be captured before the position/pose change is reflected. posing a short sleep can decrease such chances.

        # request image captures
        exposure_bias = generate_random_scalar(level_info.auto_exposure_bias_range)
        for camera_name in ["front_left", "front_right"]:
            client.simSetCameraPostProcess(
                camera_name,
                auto_exposure_bias = exposure_bias,
                auto_exposure_max_brightness = 1.0, auto_exposure_min_brightness = 1.0,
                lens_flare_intensity = 0.0,
                motion_blur_amount = 0.0,
            )

        responses = client.simGetImages(requests)

        # request attribute data


        # save images and attribute data
        for response, request in zip(responses, requests):
            img = airsim.decode_image_response(response)
            if img.dtype == np.float16:
                img = np.asarray((img * 1000).clip(0, 65535), dtype=np.uint16)  # convert to an uint16 depth image with unit mm. (we are expecting that the depth camera range is up to 65m)

            if request.image_type == airsim.ImageType.Scene:
                name = 'rgb'
            elif request.image_type == airsim.ImageType.DepthPlanar:
                name = 'depth'    # be careful that simulated depth values are not so accurate at far range due to the limited bit-depth of rengdering depth buffer.
            elif request.image_type == airsim.ImageType.Segmentation:
                name = 'mask'
            else:
                raise Exception('no impl')

            if request.camera_name == 'front_left':
                name += 'L'
            elif request.camera_name == 'front_right':
                name += 'R'
            else:
                raise Exception('no impl')

            airsim.write_png('{}{:0>8}_{}.png'.format(capture_folder, idx, name), img)

    client.simDestroyObject(item_name)
    client.simDestroyObject(human_name)


# custom info for your project
camera_info_list = {
    'zed-2': CameraInfo(HFOV=110, baseline=12),
    'zed-mini': CameraInfo(HFOV=90, baseline=6.3),
}

level_info_list = {
    'BlueprintOffice': LevelInfo(
        position_range=(-700, 700, 500, 2000, -90, -90),
        auto_exposure_bias_range=(0.0, 2.0),
    ),
    'Room': LevelInfo(
        position_range=(80, 160, -570, -200, -90, -90),
        auto_exposure_bias_range=(2.0, 5.0),
    ),
    'RoomNight': LevelInfo(
        position_range=(80, 160, -570, -200, -90, -90),
        auto_exposure_bias_range=(3.0, 6.0),
    ),
}

metahumans_hand_rotation_range = (-210, 60, -10, 30, -90, 90)
metahumans_bp_path_template = '/Game/MetaHumans/{0}/BP_{0}.BP_{0}_C'

if __name__ == '__main__':

    camera_name = 'zed-mini'
    align_to_left_camera = True
    image_width, image_height = 640, 360
    requests = [
        airsim.ImageRequest("front_left", airsim.ImageType.Scene, pixels_as_float=False),
        #airsim.ImageRequest("front_left", airsim.ImageType.DepthPlanar, pixels_as_float=True),
        #airsim.ImageRequest("front_left", airsim.ImageType.Segmentation, pixels_as_float=False),
        #airsim.ImageRequest("front_right", airsim.ImageType.Scene, pixels_as_float=False),
    ]
    level_name = 'BlueprintOffice'
    num_capture = 10

    main(
        camera_name, align_to_left_camera,
        image_width, image_height, requests,
        level_name,
        num_capture
    )