import csv
import glob
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pybullet as p
from pynput import keyboard
from tasknet.object_taxonomy import ObjectTaxonomy

import gibson2
from gibson2.external.pybullet_tools import utils
import gibson2.object_states.open as open_state
from gibson2.objects.articulated_object import URDFObject
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator
from gibson2.utils import urdf_utils
from gibson2.utils.assets_utils import download_assets

download_assets()


def get_categories():
    dir = os.path.join(gibson2.ig_dataset_path, 'objects')
    return [cat for cat in os.listdir(dir) if os.path.isdir(get_category_directory(cat))]


def get_category_directory(category):
    return os.path.join(gibson2.ig_dataset_path, 'objects', category)


def get_urdf(objdir):
    return os.path.join(objdir, os.path.basename(objdir) + ".urdf")


def get_obj(objdir):
    return URDFObject(get_urdf(objdir), name="obj", model_path=objdir)


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def main():
    # Collect the relevant categories.
    categories = get_categories()

    # Now collect the actual objects.
    objects = []
    for cat in categories:
        cd = get_category_directory(cat)
        for obj in os.listdir(cd):
            objects.append((cat, obj))

    scene_files = list(glob.glob(
        os.path.join(gibson2.ig_dataset_path, "scenes", "**", "*task*.urdf"), recursive=True))

    by_scene = {}
    by_object = {x: [] for x in objects}
    for sf in scene_files:
        tree = ET.parse(sf)

        sn = os.path.splitext(os.path.basename(sf))[0]
        scene_objs = []

        for pair in objects:
            nodes = tree.findall(".//link[@category='%s'][@model='%s']" % pair)
            if nodes:
                scene_objs.append(pair)
                by_object[pair].append(sn)

        by_scene[sn] = scene_objs

        print("%d objects in %s" % (len(scene_objs), sn))

    with open("by_object.csv", "w") as f:
        w = csv.writer(f)
        w.writerows(["%s/%s" % pair] + scenes for pair, scenes in sorted(by_object.items()))

    with open("by_scene.csv", "w") as f:
        w = csv.writer(f)
        w.writerows([scene] + ["%s/%s" % pair for pair in objects] for scene, objects in sorted(by_scene.items()))

if __name__ == "__main__":
    main()
