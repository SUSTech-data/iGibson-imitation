import logging
import os
from sys import platform

import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets

import av


def main(selection="user", headless=True, short_exec=True):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs_int (interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # If they have not been downloaded before, download assets
    download_assets()
    config_filename = os.path.join(igibson.configs_path, "turtlebot_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # output_filename = 'output.mp4'
    # fps = 30  # Frames per second
    # width = 512
    # height = 512
    # container = av.open(output_filename, mode='w')

    # stream = container.add_stream('libx264', rate=fps)
    # stream.width = width
    # stream.height = height
    # stream.pix_fmt = 'yuv420p'

    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5

    # Improving visuals in the example (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True

    # config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                # frame = av.VideoFrame.from_ndarray(state["rgb"], format='rgb')

                # # Convert the frame to YUV420 for encoding
                # frame_yuv420 = frame.reformat(format='yuv420p')

                # # Encode the frame
                # packet = stream.encode(frame_yuv420)
                # if packet:
                #     container.mux(packet)
                if done:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()
    # packet = stream.encode(None)
    # if packet:
    #     container.mux(packet)

    # # Close the container
    # container.close()
    # print("Video saved to", output_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
