import logging
import os
import sys

import cv2
import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_scene_path


import av



def main(selection="user", headless=True, short_exec=True):
    """
    Creates renderer and renders RGB images in Rs (no interactive). No physics.
    The camera view can be controlled.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    global _mouse_ix, _mouse_iy, down, view_direction

    # If a model is given, we load it, otherwise we load Rs mesh (non interactive)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path("Rs"), "mesh_z_up.obj")

    # Create renderer object and load the scene model
    renderer = MeshRenderer(width=512, height=512)

    output_filename = 'output.mp4'
    fps = 30  # Frames per second
    width = 512
    height = 512
    container = av.open(output_filename, mode='w')

    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    renderer.load_object(model_path)
    renderer.add_instance_group([0])

    # Print some information about the loaded model
    print("visual objects {}, instances {}".format(renderer.visual_objects, renderer.instances))
    print("{} {}".format(renderer.material_idx_to_material_instance_mapping, renderer.shape_material_idx))

    # Create a simple viewer with OpenCV and a keyboard navigation
    px = 0
    py = 0.2
    camera_pose = np.array([px, py, 0.5])
    view_direction = np.array([0, -1, -1])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)
    _mouse_ix, _mouse_iy = -1, -1
    down = False

    # Define the function callback for OpenCV events on the window
    def change_dir(event, x, y, flags, param):
        global _mouse_ix, _mouse_iy, down, view_direction
        if event == cv2.EVENT_LBUTTONDOWN:
            _mouse_ix, _mouse_iy = x, y
            down = True
        if event == cv2.EVENT_MOUSEMOVE:
            if down:
                dx = (x - _mouse_ix) / 100.0
                dy = (y - _mouse_iy) / 100.0
                _mouse_ix = x
                _mouse_iy = y
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
                view_direction = r1.dot(r2).dot(view_direction)
        elif event == cv2.EVENT_LBUTTONUP:
            down = False

    if not headless:
        cv2.namedWindow("Viewer")
        cv2.setMouseCallback("Viewer", change_dir)

    # Move camera and render
    max_steps = -1 if not short_exec else 500
    step = 0
    while step != max_steps:
        with Profiler("Render"):
            [render_res] = renderer.render(modes=("rgb"))   # (H, W, 4)    
            
        rgba = (render_res * 255).astype(np.uint8)

        frame = av.VideoFrame.from_ndarray(rgba, format='rgba')

        # Convert the frame to YUV420 for encoding
        frame_yuv420 = frame.reformat(format='yuv420p')

        # Encode the frame
        packet = stream.encode(frame_yuv420)
        if packet:
            container.mux(packet)

        if not headless:
            cv2.imshow("Viewer", cv2.cvtColor(np.concatenate(render_res, axis=1), cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(1)
            if q == ord("w"):
                px += 0.01
            elif q == ord("s"):
                px -= 0.01
            elif q == ord("a"):
                py += 0.01
            elif q == ord("d"):
                py -= 0.01
            elif q == ord("q"):
                break

        camera_pose = np.array([px, py, 0.5])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        step += 1

    # Cleanup
    renderer.release()
    packet = stream.encode(None)
    if packet:
        container.mux(packet)

    # Close the container
    container.close()
    print("Video saved to", output_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
