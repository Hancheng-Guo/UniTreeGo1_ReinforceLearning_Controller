import os
import shutil
import xml.etree.ElementTree as ET


def modify_model_camera(dir_original: str,
                        dir_modified: str,
                        camera_pos: str,
                        camera_xyaxes: str):

    os.makedirs(dir_modified, exist_ok=True)
    for root, dirs, files in os.walk(dir_original):
        rel_path = os.path.relpath(root, dir_original)
        dst_path = os.path.join(dir_modified, rel_path)
        os.makedirs(dst_path, exist_ok=True)
        for file in files:
            shutil.copy2(os.path.join(root, file), os.path.join(dst_path, file))

    xml_tree = ET.parse(dir_modified + "go1.xml")

    camera = None
    for cam in xml_tree.getroot().iter("camera"):
        if cam.attrib.get("name") == "tracking":
            camera = cam
            break
    if camera is None:
        raise ValueError(f"Camera 'tracking' not found in XML.")
    
    camera.set("pos", camera_pos)
    camera.set("xyaxes", camera_xyaxes)
    xml_tree.write(dir_modified + "go1.xml")