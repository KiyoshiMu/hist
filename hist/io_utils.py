from __future__ import annotations

import json
import pickle
from pathlib import Path
from zipfile import ZipFile




def pkl_load(src_p):
    with open(src_p, "rb") as target:
        out = pickle.load(target)
    return out


def pkl_dump(obj, out_p):
    with open(out_p, "wb") as target:
        pickle.dump(obj, target)


def json_dump(obj, dst_p):
    with open(dst_p, "w", encoding="utf8") as target:
        json.dump(obj, target)


def json_load(src_p):
    with open(src_p, "r", encoding="utf8") as target:
        docs = json.load(target)
    return docs


def collect_files(dir_p: str, ext=".json"):
    _dir_p = Path(dir_p)
    return _dir_p.rglob(f"*{ext}")


def mk_out_path(dst, name, mkdir=True):
    dst_p = Path(dst)
    if mkdir:
        dst_p.mkdir(exist_ok=True)
    return dst_p / name


def unzip_file(src, dst):
    Path(dst).mkdir(exist_ok=False)
    with ZipFile(src, "r") as in_f:
        in_f.extractall(dst)


def simplify_label(labels: list[str]|str) -> str:
    if isinstance(labels, str):
        return labels
    if "Myelodysplastic syndrome" in labels:
        return "MDS"
    if 'Normal' in labels:
        return "NORMAL"
    if 'Plasma cell neoplasm' in labels:
        return "PCN"
    if 'Acute leukemia' in labels:
        return "ACL"
    if 'Lymphoproliferative disorder' in labels:
        return "LPD"
    if 'Myeloproliferative neoplasm' in labels:
        return 'MPN'
    return "OTHER"