import os
import shutil
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree


def modify_model(dir_original: str, dir_modified: str, **kwargs):
    os.makedirs(dir_modified, exist_ok=True)
    for root, dirs, files in os.walk(dir_original):
        rel_path = os.path.relpath(root, dir_original)
        dst_path = os.path.join(dir_modified, rel_path)
        os.makedirs(dst_path, exist_ok=True)
        for file in files:
            shutil.copy2(os.path.join(root, file), os.path.join(dst_path, file))
    modify_go1(dir=dir_modified, **kwargs)
    modify_scene(dir=dir_modified, **kwargs)
    print("Model modification completed!")

def modify_go1(dir: str, **kwargs):
    file = os.path.join(dir, "go1.xml")
    xml_tree = ET.parse(file)
    modify_go1_camera(xml_tree=xml_tree, **kwargs)
    modify_go1_foot(xml_tree=xml_tree, **kwargs)
    xml_tree.write(file)

def modify_go1_camera(xml_tree: ElementTree, camera_pos: str, camera_xyaxes: str, **kwargs):
    camera = None
    for cam in xml_tree.getroot().iter("camera"):
        if cam.attrib.get("name") == "tracking":
            camera = cam
            break
    if camera is None:
        raise ValueError(f"Camera 'tracking' not found in XML.")
    camera.set("pos", camera_pos)
    camera.set("xyaxes", camera_xyaxes)

def modify_go1_foot(xml_tree: ElementTree, friction: str="1.8 0.1 0.001", **kwargs):
    foot_name = ["FL", "FR", "RL", "RR"]
    for geom in xml_tree.getroot().iter("geom"):
        if geom.attrib.get("name") in foot_name:
            geom.set("friction", friction)

def modify_scene(dir: str, **kwargs):
    file = os.path.join(dir, "scene.xml")
    xml_tree = ET.parse(file)
    modify_scene_floor(xml_tree=xml_tree, **kwargs)
    xml_tree.write(file)

def modify_scene_floor(xml_tree: ElementTree, friction: str="1.5 0.1 0.001", **kwargs):
    floor = None
    for geom in xml_tree.getroot().iter("geom"):
        if geom.attrib.get("name") == "floor":
            floor = geom
            break
    if floor is None:
        raise ValueError(f"Floor not found in XML.")
    floor.set("friction", friction)
    