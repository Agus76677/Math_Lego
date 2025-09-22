# tools/gen_tree_md.py  — UTF-8 safe, Markdown tree with clickable links
from __future__ import annotations
import os, argparse

IGNORE_DIRS  = {'.git', '__pycache__', '.ipynb_checkpoints', '.idea', '.vscode'}
IGNORE_FILES = {'.DS_Store'}

def human_size(n:int)->str:
    for u in ['B','KB','MB','GB','TB']:
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}PB"

def gen_md(root:str, show_size:bool=True, max_depth:int|None=None)->str:
    root = os.path.normpath(root)
    lines = [f"# Directory tree: `{root}`\n"]
    for cur, dirs, files in os.walk(root, topdown=True):
        # 过滤
        dirs[:]  = [d for d in sorted(dirs)  if d not in IGNORE_DIRS]
        files    = [f for f in sorted(files) if f not in IGNORE_FILES]

        rel = os.path.relpath(cur, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if max_depth is not None and depth > max_depth:
            continue

        if rel != ".":
            indent = "  " * (rel.count(os.sep))
            lines.append(f"{indent}- **{os.path.basename(cur)}/**")

        indent = "  " * (0 if rel=="." else rel.count(os.sep)+1)
        for f in files:
            p  = os.path.join(cur, f)
            rp = os.path.relpath(p, root).replace("\\", "/")
            size = f" ({human_size(os.path.getsize(p))})" if show_size else ""
            lines.append(f"{indent}- [{f}]({rp}){size}")
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("-o", "--out", default="DIRECTORY_TREE.md")
    ap.add_argument("--no-size", action="store_true")
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    md = gen_md(args.path, show_size=not args.no_size, max_depth=args.max_depth)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print("Written:", args.out)
