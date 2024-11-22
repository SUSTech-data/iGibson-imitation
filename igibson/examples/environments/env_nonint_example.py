import logging
import os
from sys import platform

import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets, download_demo_data

import numpy as np
import av
import cv2


def main(selection="user", headless=True, short_exec=False):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs (not interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # If they have not been downloaded before, download assets and Rs Gibson (non-interactive) models
    download_assets()
    download_demo_data()
    config_filename = os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    output_filename = 'output.mp4'
    fps = 30  # Frames per second
    width = 640
    height = 480
    container = av.open(output_filename, mode='w')

    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5

    # Shadows and PBR do not make much sense for a Gibson static mesh
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False

    env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)

                # [rgb, depth, seg, ins_seg] = env.simulator.renderer.render(modes=('rgb','3d','seg', 'ins_seg'))
                # seg = (seg[:, :, 0:1] * MAX_CLASS_COUNT).astype(np.int32)
                # ins_seg = (ins_seg[:, :, 0:1] * MAX_INSTANCE_COUNT).astype(np.int32)
                # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                frame = av.VideoFrame.from_ndarray((obs["rgb"]*255).astype(np.uint8), format='rgb24')

                # Convert the frame to YUV420 for encoding
                frame_yuv420 = frame.reformat(format='yuv420p')

                # Encode the frame
                packet = stream.encode(frame_yuv420)
                if packet:
                    container.mux(packet)

                if done:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()
    packet = stream.encode(None)
    if packet:
        container.mux(packet)

    # Close the container
    container.close()
    print("Video saved to", output_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
