from ctypes import *
from source.core.config import cfg_priv
import numpy as np
from numpy.ctypeslib import ndpointer
import logging


class HungarianAlgorithm(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(eval(cfg_priv.LOG))
        self.lib = cdll.LoadLibrary(cfg_priv.GLOBAL.VISION_PROJECT_ROOT + '/source/modules/fh_tracking/Hungarian/libHungarian.so')
        self.lib.init_Hungarian.restype = c_void_p
        self.obj = self.lib.init_Hungarian()

    def Solve(self,DistMatrix, row, col):
        DistMatrix_flatten = sum(DistMatrix,[])
        DistMatrix_flatten_num = row * col
        DistMatrix_flatten = (c_float * DistMatrix_flatten_num)(*DistMatrix_flatten)
        self.lib.Solve.argtypes = [c_void_p, (c_float * (DistMatrix_flatten_num)), c_int, c_int]
        self.lib.Solve.restype = ndpointer(dtype=c_int, shape=(col,))
        Solve_result = self.lib.Solve(self.obj, DistMatrix_flatten, row, col)
        Solve_result = Solve_result.astype(np.int32).tolist()
        return Solve_result
