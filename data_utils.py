import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision import datasets, transforms
from tqdm import tqdm
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset, Dataset
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


def split_dataset(ds, split=0.1):
    """
    split dataset into 2 datasets according to split ratio
    """
    num_train = len(ds)
    split = int(num_train * split)
    indices = list(range(num_train))

    train_idx, val_idx = indices[split:], indices[:split]
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    return train_ds, val_ds

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda x, y: (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

class pairs_dataset(Dataset):
    """
    create a dataset object from 2 detesets.
    if ds1 has tuples of (x1,y1) and ds2 has tuples of (x2,y2)
    new dataset will have (x1, x2, y2)
    """
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        assert len(ds1) == len(ds2)

    def __getitem__(self, idx):
        return self.ds1[idx][0], self.ds2[idx][0], self.ds2[idx][1]

    def __len__(self):
        return len(self.ds1)


def adversarialize(dataset, torch_model):
    """
    get a Dataset object and a PyTorch model and return an adversarial version of that dataset

    dataset: a Dataset object (note: not a DataLoader!)
    torch_model: a PyTorch model

    returns a Dataset object of the adversarial dataset
    """
    # We use tf for evaluation on adversarial data
    data_shape = dataset[0][0].shape
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, *data_shape))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(torch_model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    # Create an FGSM attack
    fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
    fgsm_params = {'eps': 0.1,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
    adv_preds_op = adv_x_op

    # Run an evaluation of our model against fgsm
    print("==> creating an adversarial version of dataset...")
    adv_xs = []
    ys = []
    for i, (x, y) in tqdm(enumerate(dataset), total=len(dataset)):
        x = x.unsqueeze(0)
        x = x.to('cpu')
        adv_x = sess.run(adv_preds_op, feed_dict={x_op: x})
        adv_xs.append(adv_x)
        ys.append(torch.LongTensor([y]))

    tensor_x = torch.cat([torch.Tensor(i) for i in adv_xs])
    tensor_y = torch.cat(ys)
    adv_dataset = utils.TensorDataset(tensor_x, tensor_y)
    print(f"==> {len(adv_dataset)} examples")
    return adv_dataset
