import os

from tool.path_helper import ROOT_DIR

excludes_list = []

for filename in os.listdir(ROOT_DIR.joinpath("out/checkpoints")):
    excluded = False
    for name in excludes_list:
        if name in filename:
            excluded = True
    if not excluded:
        os.remove(ROOT_DIR.joinpath("out/checkpoints/" + filename))
