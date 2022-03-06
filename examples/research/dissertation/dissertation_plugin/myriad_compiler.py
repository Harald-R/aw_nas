# -*- coding: utf-8 -*-

import io
import logging
import os
import pickle
import re
import sys
import shutil
import subprocess
from collections import namedtuple

import yaml
import torch
from torch.autograd import Variable

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.main import _init_component
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.hardware.base import BaseHardwareCompiler
from aw_nas.utils.log import LEVEL as _LEVEL
from aw_nas.rollout.general import GeneralSearchSpace
from aw_nas.hardware.utils import Prim, assemble_profiling_nets

try:
    from aw_nas.utils.pytorch2caffe import pytorch_to_caffe
except:
    pytorch_to_caffe = None

AWNAS_PATH = '/home/harald/git/aw_nas'
MO_PATH = '/home/harald/git/openvino/model-optimizer/mo.py'
COMPILER_PATH = '/home/harald/git/openvino/bin/intel64/Debug/myriad_compile'

CAFFE_DATA_LAYER_STR = """
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    crop_size: INPUT_SIZE
    mean_value: 103.53
    mean_value: 116.28
    mean_value: 123.675
    use_standard_std: true
  }
  image_data_param {
    source: "/datasets/imgNet/imagenet1k_valid_source.txt"
    root_folder: "/datasets/imgNet/imagenet1k_valid_dataset/"
    batch_size: 50
    new_height: 256
    new_width: 256
  }
}
"""


class MyriadCompiler(BaseHardwareCompiler):
    """
    A hardware interface class for Myriad X (CNN accelerator).
    """

    NAME = "myriad"

    def __init__(self, dcf=None, mode="debug", calib_iter=0, gpu=0, input_size=None):
        super(MyriadCompiler, self).__init__()

        expect(input_size is not None, "must specificy `input_size`", ConfigException)
        self.dcf = dcf
        self.mode = mode
        self.calib_iter = calib_iter
        self._debug_output = _LEVEL <= logging.DEBUG  # debug output
        self.gpu = gpu
        self.input_size = input_size

    def compile(self, compile_name, net_cfg, result_dir):
        search_space = _init_component(net_cfg, "search_space")
        assert isinstance(search_space, GeneralSearchSpace)
        model = _init_component(
            net_cfg,
            "final_model",
            search_space=search_space,
            device="cpu",
        )

        onnx_model_path = '/tmp/model.onnx'
        input_size = [1, 3] + list(self.input_size)
        torch.onnx.export(model, torch.rand(input_size), onnx_model_path)

        openvino_model_dir = '/tmp/IR'
        os.system('python3 {} --input_model {} --output_dir {}'
                    .format(MO_PATH, onnx_model_path, openvino_model_dir))

        openvino_model_path = os.path.join(openvino_model_dir, 'model.xml')
        blob_path = os.path.join(openvino_model_dir, 'model.blob')
        os.system('{} -m {} -o {}'
                    .format(COMPILER_PATH, openvino_model_path, blob_path))
        # import ipdb; ipdb.set_trace()

        return blob_path

    def parse_file(
        self,
        prof_result_dir,
        prof_prim_file,
        prim_to_ops_file,
        result_dir,
        perf_fn=None,
        perf_names=("latency",),
    ):
        # prof_result consists of all basic operators ,like conv_bn_relu, pooling and concat
        """
        prof_result_dir: the measurement result file
        prof_prim_file: all primitives that need to be profiled.
        prim_to_ops_file: primitive to the names of caffe layers file
        """

        # load prims to op dict
        prim_to_ops = dict()
        for _dir in os.listdir(prim_to_ops_file):
            cur_dir = os.path.join(prim_to_ops_file, _dir, 'pytorch_to_caffe')
            for _file in os.listdir(cur_dir):
                if not _file.endswith('prim2names.pkl'):
                    continue
                pkl = os.path.join(cur_dir, _file)
                with open(pkl, "rb") as r_f:
                    _dict = pickle.load(r_f)
                    prim_to_ops.update(_dict)
                    self.logger.info("Unpickled file: {pkl}".format(pkl=pkl))

        # meta info: prim_to_ops is saved when generate profiling final-yaml folder for each net
        for _dir in os.listdir(prof_result_dir):
            cur_dir = os.path.join(prof_result_dir, _dir)
            if os.path.isdir(cur_dir):
                parsed_dir = os.path.join(result_dir, _dir)
                os.makedirs(parsed_dir, exist_ok=True)
                for i, _file in enumerate(os.listdir(cur_dir)):
                    perf_yaml = self.parse_one_network(
                        os.path.join(cur_dir, _file), prof_prim_file, prim_to_ops)
                    if perf_yaml:
                        with open(os.path.join(parsed_dir, i + ".yaml"),
                                "w") as fw:
                            yaml.safe_dump(perf_yaml, fw)

    def parse_one_network(
        self,
        prof_result_file,
        prof_prim_file,
        prim_to_ops,
        perf_fn=None,
        perf_names=("latency",)
    ):
        # parse result file
        profiled_yaml = {"overall_{}".format(k): 0. for k in perf_names}
        if perf_fn is None:
            perf_fn = lambda split_line: (
                split_line[0],
                {"latency": float(split_line[3])},
            )
        name_to_perf_dict = {}
        Perf = namedtuple("Perf", perf_names)
        with open(prof_result_file, "r") as fl:
            fl_lines = fl.readlines()
            for line in fl_lines[3:-3]:
                if "Total Time" in line:
                    profiled_yaml["overall_latency"] = int(re.findall(r"(\d+)", a[-5])[0])
                    continue
                split_line = re.split(r"\s+", line)
                if len(split_line) > 4:
                    name, performance = perf_fn(split_line)
                    if not name.startswith("NodeName"):
                        if name not in name_to_perf_dict:
                            name_to_perf_dict[name] = Perf([], [])
                        for perf in perf_names:
                            getattr(name_to_perf_dict[name], perf).append(
                                performance[perf]
                                    )
        name_to_perf_dict = {
            k: Perf(
                *[sum(getattr(v, perf)) / len(getattr(v, perf)) for perf in perf_names]
            )
            for k, v in name_to_perf_dict.items()
        }
        # mapping name of op to performances
        # {conv2: Perf(latency=12., memory: 2048), ...}

        with open(prof_prim_file, "r") as fr:
            prof_prim = yaml.load(fr)

        nets_prim_to_perf = []
        for prim in prof_prim:
            # prim consists of prim_type, input_ch, output_ch, kernel, stride
            # Now, use prim_to_ops mapping prim into basic ops' names
            # Using function instead of dict to handle exceptions
            names = prim_to_ops(Prim(**prim))
            if len(set.intersection(names, name_to_perf_dict)) == 0:
                self.logger.debug("prims {} is not measured.".format(prim))
                continue
            for perf in perf_names:
                prim[perf] = sum(
                    [
                        getattr(name_to_perf_dict[name], perf)
                        for name in names
                        if name in name_to_perf_dict
                    ]
                )
            nets_prim_to_perf.append(prim)
        profiled_yaml["primitives"] = nets_prim_to_perf
        return profiled_yaml

if __name__ == '__main__':
    hwobj_cfg_file = '{}/examples/hardware/configs/ofa_lat.yaml'.format(AWNAS_PATH)
    cfg_file = '{}/examples/hardware/configs/ofa_final.yaml'.format(AWNAS_PATH)
    with open(cfg_file, "r") as ss_cfg_f:
        ss_cfg = yaml.load(ss_cfg_f, Loader=yaml.FullLoader)
    with open(hwobj_cfg_file, "r") as hw_cfg_f:
        hw_cfg = yaml.load(hw_cfg_f, Loader=yaml.FullLoader)

    ss = get_search_space(hw_cfg["mixin_search_space_type"],
                          **ss_cfg["search_space_cfg"],
                          **hw_cfg["mixin_search_space_cfg"])

    assert 'prof_prims_cfg' in hw_cfg, "key prof_prims_cfg must be specified in hardware configuration file."
    hw_obj_cfg = hw_cfg['prof_prims_cfg']
    prof_prims = list(
        ss.generate_profiling_primitives(**hw_obj_cfg))

    # prof_prims = {'prim_type': 'mobilenet_v3_block', 'spatial_size': 19, 'C': 96, 'C_out': 96, 'stride': 1, 'affine': True, 'activation': 'h_swish', 'expansion': 3, 'kernel_size': 5, 'use_se': True}
    prof_net_cfgs = assemble_profiling_nets(prof_prims,
                                                **hw_cfg["profiling_net_cfg"])
    prof_cfg = list(prof_net_cfgs)[0]

    compiler = MyriadCompiler(input_size=(32,32))
    compiler.compile(compile_name='model', net_cfg=prof_cfg, result_dir='results')
