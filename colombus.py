import os
import os.path as osp
from pathlib import Path
from loguru import logger
import psutil
from datetime import datetime
import pandas as pd
import yaml
from glob import glob


TIMESTEP_COL = 'timestamp'


def summarize_metrics_df(df, columns=None):
    out = {}
    cols = df.columns if columns is None else columns

    for col in cols:
        # get only number values
        number_idx = df[col].apply(lambda x: isinstance(x, (int, float, complex)) and not isinstance(x, bool))
        data = df[col][number_idx]
        data = data.dropna()
        if len(data) > 0:
            out[col] = {
                "mean": df[col].mean(),
                "max": df[col].max()}
    return pd.DataFrame(out)


class Experiment:
    def __init__(self, base_path, save_every=1):
        self.pid = os.getpid()

        self.folder = Path(base_path) / str(self.pid)
        os.makedirs(self.folder, exist_ok=True)

        self.ckpt_path = self.folder / "ckpt"
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.metrics_path = self.folder / "metrics.csv"
        self.metrics = []

        self.hparams_path = self.folder / "hparams.yaml"
        self.hparams = {}

        self.log_steps = 0
        self.save_every = save_every

        self.cmd = " ".join(psutil.Process(self.pid).cmdline())
        with open(self.folder / "run.sh", "w") as f:
            f.write(self.cmd)

        logger.info(f"process ID: {self.pid}")
        logger.info(f"process command: {self.cmd}")
        logger.info(f"run folder: {self.folder}")

    def log_metrics(self, metrics):
        self.metrics.append({**metrics, **{TIMESTEP_COL: str(datetime.utcnow())}})

        self.log_steps += 1
        if self.log_steps % self.save_every == 0:
            self.save()

    def log_hyperparameters(self, hparams):
        self.hparams = hparams
        with open(self.hparams_path, 'w') as f:
            yaml.dump(self.hparams, f, default_flow_style=False)

    def save(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_path, index=False)

    def print_metrics(self):
        df = pd.DataFrame(self.metrics)
        print(summarize_metrics_df(df))


class Monitor:
    def __init__(self, base_path, metrics):
        self.base_path = base_path
        self.metrics = metrics

    def fetch(self):
        experiments = [f.path for f in os.scandir(self.base_path) if f.is_dir()]

        summary = []
        for exp in experiments:
            metrics_path = osp.join(exp, "metrics.csv")
            exp_summary = {"path": exp}
            try:
                df = pd.read_csv(metrics_path)[self.metrics]
                for m in self.metrics:
                    exp_summary[f'min_{m}'] = df[m].min()
                    exp_summary[f'max_{m}'] = df[m].max()
            except: pass
            summary.append(exp_summary)
        print(pd.DataFrame(summary))



if __name__ == "__main__":
    # exp = Experiment("/tmp/project")
    # exp.log_metrics({'loss': 2, 'acc': 40})
    # exp.log_metrics({'loss': 1, 'acc': 70})
    # exp.log_metrics({'loss': 9.5})
    # exp.log_metrics({'score': 100})
    # exp.log_hyperparameters({'lr': 0.0001, 'batch_size': 8})
    # exp.print_metrics()

    mon = Monitor("/tmp/project", ["loss", "acc"])
    mon.fetch()