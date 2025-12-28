import json
import os
import re

INLINE_KEYS = {
    "atom_idx",
    # "core_atoms",
    # "attachment_atoms",
    # "substituent_indices",
}

INLINE_PREFIX = "__INLINE__"
INLINE_SUFFIX = "__INLINE__"


def _mark_inline_lists(obj):
    """
    递归遍历 dict / list
    对指定 key 下的 list 打标记
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k in INLINE_KEYS and isinstance(v, list):
                new[k] = f"{INLINE_PREFIX}{json.dumps(v)}{INLINE_SUFFIX}"
            # 支持以"_list"或"_indices"结尾的key
            # elif k.endswith("_indices") and isinstance(v, list):
            #     new[k] = f"{INLINE_PREFIX}{json.dumps(v)}{INLINE_SUFFIX}"
            else:
                new[k] = _mark_inline_lists(v)
        return new

    elif isinstance(obj, list):
        return [_mark_inline_lists(x) for x in obj]

    return obj


def dump_json_pretty(data, path, indent=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    marked = _mark_inline_lists(data)

    with open(path, "w") as f:
        json.dump(marked, f, indent=indent, ensure_ascii=False)

    # post-process：把标记字符串变回真正数组
    with open(path) as f:
        text = f.read()

    text = re.sub(
        rf'"{INLINE_PREFIX}(\[.*?\]){INLINE_SUFFIX}"',
        r"\1",
        text,
    )

    with open(path, "w") as f:
        f.write(text)
