import time
import spur
import itertools
import os
import os.path as osp
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse
from argparse import Namespace
import copy
from collections import namedtuple
import uuid
from shutil import copytree, ignore_patterns

Node = namedtuple('Node', "name connection n_gpus")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--conf', default='conf/config.yaml', help='path to base configuration yaml. this will contain the default hyperparameters')
parser.add_argument('--grid', default='grid/grid.yaml', help='path to base grid yaml. this will contain the the grid override hyperparameters')
args = parser.parse_args()

def run():

    # get the default hyerparams
    base_settings = OmegaConf.to_container(OmegaConf.load(args.conf))
    if 'hydra' in base_settings: del base_settings['hydra']

    # get grid settings
    grid_yaml      = OmegaConf.to_container(OmegaConf.load(args.grid))
    python_bin     = grid_yaml['settings']['python']
    ts             = grid_yaml['settings']['ts']
    project_dir    = grid_yaml['settings']['wd']
    target_wd      = grid_yaml['settings']['tmp_wd']
    grids          = grid_yaml['grids']
    commons        = grid_yaml['common']
    nodes_settings = grid_yaml['nodes']

    # copy code to a new directory so future code editing will not effect a running grid
    target_wd = osp.join(target_wd, f"grid_{uuid.uuid4().hex}")
    copytree(project_dir, target_wd, ignore=ignore_patterns(".git"))
    print(f"copied code to: {target_wd}")

    # connect to all nodes and get n_gpus
    nodes = []
    for node in nodes_settings:
        conn = spur.SshShell(hostname=node, username="XXX", password="XXX", missing_host_key=spur.ssh.MissingHostKey.accept)
        n_gpus = int(conn.run(['nvidia-smi', '-L']).output.decode("utf-8").count('\n'))
        nodes.append(Node(name=node, connection=conn, n_gpus=n_gpus))
    n_nodes = len(nodes)
    
    # send all commands for grid to nodes
    run_id = 0
    gpu_id = 0
    node_id = 0
    for grid in grids:
        # get default parameters in 'cur_cfg'
        grid = {**grid, **commons}
        grid = {k: v if type(v) == list else [v] for k, v in grid.items()}
        cur_cfg = copy.deepcopy(base_settings)
        cartesian_product = (dict(zip(grid, x)) for x in itertools.product(*grid.values()))

        # override default parameters with the grid parameters
        for sub_exp_idx, combination in enumerate(cartesian_product):
            override_name = ""
            for k, v in combination.items():
                cur_cfg[k] = v
                override_name += f",{k}={v}"
            # set unique name
            cur_cfg['exp_name'] = f"'{str(run_id)}{override_name}'"
            # finalize the command
            args_str  = f"{ts} {python_bin} main.py"
            for k, v in cur_cfg.items():
                args_str += f" {k}={v}"
            
            env = {
                'USE_SIMPLE_THREADED_LEVEL3': '1',
                'OMP_NUM_THREADS': '1',
                'TS_SOCKET': f"/tmp/felix_gpu_{gpu_id}",
                'CUDA_VISIBLE_DEVICES': f"{gpu_id}"
            }
            nodes[node_id].connection.run(args_str.split(" "), cwd=target_wd, update_env=env)

            run_id += 1
            gpu_id +=1
            if gpu_id > nodes[node_id].n_gpus - 1:
                gpu_id = 0
                node_id = (node_id + 1) % n_nodes

        print(f"ran a grid of size: {sub_exp_idx + 1}")
    print(f"\noverall: {run_id}")

if __name__ == "__main__":
    run()
