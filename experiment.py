import subprocess
import atexit
import pandas as pd
from datetime import datetime
from tensorboardX import SummaryWriter
from loguru import logger
import yaml
import argparse
from argparse import Namespace
import time
from tensorboardX import SummaryWriter
import os
from os.path import join
from distutils.dir_util import copy_tree


def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


class Experiment(object):
    def __init__(self, root_dir):
        self.start_time = time.time()
        self.dir = root_dir
        self.ckpt_dir = join(self.dir, 'ckpt')
        self.code_dir = join(self.dir, f'code_{get_git_hash()}')
        self.tb_dir = join(self.dir, 'tensorboard')
        self.hparams_file = join(self.dir, 'hparams.yaml')
        self.tb_writer = SummaryWriter(self.tb_dir)
        self.metrics = []
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        copy_tree(os.path.abspath("."), self.code_dir)        
        logger.info(f"experiment folder: {self.dir}")
        atexit.register(self.save)

    def save_hparams(self, hparams, hparams_file=None):
        if not hparams_file:
            hparams_file = self.hparams_file

        if type(hparams) == argparse.Namespace:
            logger.info("parsing ArgumentParser hparams") 
            hparams = vars(hparams)
        elif type(hparams) == dict:
            logger.info("parsing dict hparams") 
            pass
        else:
            logger.error("hparams type is not supported")
            return

        with open(hparams_file, "w") as f:
            f.write(yaml.dump(hparams))
            logger.info(f"hparams file saved to: {hparams_file}")

        for k,v in hparams.items():
            logger.info("\t{} - {}".format(k, v))

    def load_hparams(self, hparams_file=None):
        if not hparams_file:
            hparams_file = self.hparams_file
        logger.info(f"loading hparams from: {hparams_file}")

        hparams = yaml.load(hparams_file)
        hparams = Namespace(**hparams)
        return hparams

    def tensorboard_call(self, func, *args, **kwargs):
        func = getattr(self.tb_writer, func)
        func(*args, **kwargs)

    def log_metric(self, metrics_dict):
        # log in tensorboard
        for k,v in metrics_dict.items():
            self.tb_writer.add_scalar(k, v)

        # create timestamp
        if 'timestamp' not in metrics_dict:
            metrics_dict['timestamp'] = str(datetime.utcnow())
        self.metrics.append(metrics_dict)

    def save(self):
        logger.info("saving experiment")
        # save metrics to csv
        df = pd.DataFrame(self.metrics)
        df.to_csv(join(self.dir, 'metrics.csv'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--augment', default=True, type=bool)
    args = parser.parse_args()
    
    exp = Experiment('/tmp/exp')
    exp.save_hparams(args)
    exp.log_metric({'metrics/loss': 0.5})
    exp.log_metric({'metrics/loss': 0.4, 'metrics/acc': 0.99})
