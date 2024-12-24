import sys

sys.path.append("./src")

import os
import glob
import h5py

import numpy as np
import torch
from torch import nn

from data_modules.scene_centric import SceneCentricPreProcessing
from data_modules.sc_relative import SceneCentricRelative


n_agent = 64
n_step = 91
n_step_history = 11
n_agent_no_sim = 256
n_pl = 1024
n_tl = 100
n_tl_stop = 40
n_pl_node = 20
tensor_size_train = {
    # agent states
    "agent/valid": (n_step, n_agent),  # bool,
    "agent/pos": (n_step, n_agent, 2),  # float32
    # v[1] = p[1]-p[0]. if p[1] invalid, v[1] also invalid, v[2]=v[3]
    "agent/vel": (n_step, n_agent, 2),  # float32, v_x, v_y
    "agent/spd": (n_step, n_agent, 1),  # norm of vel, signed using yaw_bbox and vel_xy
    "agent/acc": (n_step, n_agent, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
    "agent/yaw_bbox": (n_step, n_agent, 1),  # float32, yaw of the bbox heading
    "agent/yaw_rate": (n_step, n_agent, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
    # agent attributes
    "agent/type": (n_agent, 3),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
    "agent/cmd": (n_agent, 8),  # bool one_hot
    "agent/role": (n_agent, 3),  # bool [sdc=0, interest=1, predict=2]
    "agent/size": (n_agent, 3),  # float32: [length, width, height]
    "agent/goal": (n_agent, 4),  # float32: [x, y, theta, v]
    "agent/dest": (n_agent,),  # int64: index to map n_pl
    # map polylines
    "map/valid": (n_pl, n_pl_node),  # bool
    "map/type": (n_pl, 11),  # bool one_hot
    "map/pos": (n_pl, n_pl_node, 2),  # float32
    "map/dir": (n_pl, n_pl_node, 2),  # float32
    "map/boundary": (4,),  # xmin, xmax, ymin, ymax
    # traffic lights
    "tl_lane/valid": (n_step, n_tl),  # bool
    "tl_lane/state": (n_step, n_tl, 5),  # bool one_hot
    "tl_lane/idx": (n_step, n_tl),  # int, -1 means not valid
    "tl_stop/valid": (n_step, n_tl_stop),  # bool
    "tl_stop/state": (n_step, n_tl_stop, 5),  # bool one_hot
    "tl_stop/pos": (n_step, n_tl_stop, 2),  # x,y
    "tl_stop/dir": (n_step, n_tl_stop, 2),  # x,y
}


def main():
    # Prepare data sample
    listfiles = sorted(glob.glob(os.path.join("./h5_womd_data/datasets/training_files", "*.h5")))
    for idx, fn in enumerate(listfiles):
        if "486eea22901338df" in fn:
            break
    print(listfiles[idx])

    pre_processing = [
        SceneCentricPreProcessing(
            gt_in_local=True,
            mask_invalid=False,
            time_step_current=10,
            data_size=tensor_size_train
        ),
        SceneCentricRelative(
            time_step_current=10,
            data_size=tensor_size_train,
            dropout_p_history=-1,
            use_current_tl=True,
            add_ohe=True,
            pl_aggr=False,
            pose_pe={"agent": "mpa_pl", "map": "mpa_pl"}
        )
    ]
    pre_processing = nn.Sequential(*pre_processing)
    pre_processing.train()
    
    data = {}
    with h5py.File(listfiles[idx], "r", libver="latest", swmr=True) as hf:
        for key in tensor_size_train:
            _val = np.ascontiguousarray(hf["0"][key])
            # if key.startswith("agent") and _val.shape[0] == 91:
            #     _val = np.swapaxes(_val, 0, 1)
            data[key] = _val
        print(hf["0"].attrs["scenario_center"])
        print(hf["0"].attrs["scenario_yaw"])

    data_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in data.items()}
    # print(data["agent/pos"][0, :, 0])
    # print(pre_processing)
    _batch = pre_processing(data_t)
    # print(list(data))
    # print(data["agent/role"])
    # print(list(_batch))
    batch = {}
    for k in _batch:
        if "input" in k:
            batch[k] = _batch[k].squeeze(0).numpy()
    vel = batch["input/agent_attr"][..., 7:9]
    print("here 104")
    # print(batch["input/agent_attr"][0, :, :2])
    print(batch["input/agent_attr"][0, :, 2:4])
    return


if __name__ == "__main__":
    main()
