#!/usr/bin/env python3

import numpy as np
import sys
from org.mk.training.dl.common import softmax
from org.mk.training.dl.common import softmax_grad
from org.mk.training.dl.core import Dense
from org.mk.training.dl.common import WeightsInitializer
from org.mk.training.dl import init_ops
import inspect
import os

WORK_HOME = os.environ['WORK_HOME']
TMP_DIR="resources/tmp/"
ABS_TMP_DIR= WORK_HOME+"/resources/tmp/"



def get_rel_save_dir(projname):
    firstscript=inspect.getsourcefile(sys._getframe(2))
    filename=os.path.basename(firstscript)
    filenamewe, file_extension = os.path.splitext(filename)
    savepath=str(TMP_DIR+projname+"/"+filenamewe+"/")
    make_dir_ifnot_exist(savepath)
    return savepath

def get_rel_save_file(projname):
    sfn=str(get_rel_save_dir(projname)+"model-checkpoint-")
    return sfn



def make_dir_ifnot_exist(abspath):
    if not os.path.isdir(abspath):
            os.makedirs(abspath)

