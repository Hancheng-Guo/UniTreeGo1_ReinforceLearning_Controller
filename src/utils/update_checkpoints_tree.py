import json
import os
import re
from anytree import Node, RenderTree, find


def from_dict(json_dict, parent=None):
    node = Node(json_dict["name"], parent=parent, note=json_dict["note"])
    for child in json_dict["children"]:
        from_dict(child, node)
    return node


def to_dict(node):
    json_dict = {
        "name": node.name,
        "note": node.note,
        "children": [to_dict(c) for c in node.children],
        }
    return json_dict


def to_txt(root, txt_path="", checkpoints_path=""):
    with open(txt_path, "w", encoding="utf-8") as f:
        for pre, fill, node in RenderTree(root):
            pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$')
            m = pattern.match(node.name)
            if m:
                zip_exist = os.path.isfile(
                    os.path.join(checkpoints_path, m.group(1), f"mdl_{node.name}.zip"))
                pkl_exist = os.path.isfile(
                    os.path.join(checkpoints_path, m.group(1), f"env_{node.name}.pkl"))
                marker = "" if (zip_exist and pkl_exist) else "*"
            else:
                marker = "*"
            print(f"{pre}{node.name}{marker}\t({node.note})", file=f)
        print("\n\n\nNote: Models marked with * have been deleted.", file=f)


def update_checkpoints_tree(child, parent="root", note="",
                            file_path="", checkpoints_path="."):
    json_path = f"{file_path}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    root = from_dict(data)

    node = find(root, lambda n: n.name == parent)
    if node:
        Node(child, parent=node, note=note)
    else:
        Node(child, parent=root, note=note)

    json_dict = to_dict(root)
    json.dump(json_dict, open(json_path, "w"), ensure_ascii=False, indent=2)
    to_txt(root, txt_path=f"{file_path}.txt", checkpoints_path=checkpoints_path)
