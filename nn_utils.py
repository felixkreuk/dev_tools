import torch
import torch.nn as nn


def _get_conv_output_size(shape, cnn):
    input = torch.rand(1, *shape)
    output = cnn(input)
    return output.data.view(1, -1).size(1)


def select_activation(activation):
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'leacky_relu':
        activation = nn.LeakyReLU()
    else:
        activation = None
    return activation


def cnn_layers(structure, conv_type='2d', pool=None, activation=None, last_activation=None, bn=True):
    """
    structure: list of tuples, each tuple is of the form (in_channels, out_channels, kernel_size, padding_size, dilation_size)
    conv_type: '2d' or '1d' (default is 2d)
    pool: int (e.g. 2 for 2x2 pooling), pooling type (1d/2d) is according to conv_type
    activation: type of activation function to use (None for no activation)
    last_activation: type of activation for the last layer (None for no activation)
    bn: True/False, use batch-norm or not

    returns a Sequential module with layers of CNN
    """

    # define conv type
    conv = nn.Conv2d if conv_type == '2d' else nn.Conv1d

    # define batch-norm type
    if bn:
        bn = nn.BatchNorm2d if conv_type == '2d' else nn.BatchNorm1d

    # define pooling according to conv_type
    if pool:
        pool = nn.MaxPool2d(pool) if conv_type == '2d' else nn.MaxPool1d(pool)

    # define activation
    activation = select_activation(activation)

    # build the cnn
    cnn = []
    for i, (in_c, out_c, kernel, stride, padding, dilation) in enumerate(structure):
        cnn.append(conv(in_c, out_c, kernel, stride, padding, dilation))
        if pool:
            cnn.append(pool)
        if i == len(structure)-1 and last_activation:
            cnn.append(select_activation(last_activation))
        elif activation and not (i == len(structure)-1):
                cnn.append(activation)
        if bn:
            cnn.append(bn(out_c))

    return nn.Sequential(*cnn)


def fc_layers(structure, activation=None, bn=None, last_activation=None):
    """
    returns a Sequential object according to structure

    structure: list of tuples, each tuple is of the form (in_dim, out_dim)
    activation: type of activation function to use (None for no activation)
    last_activation: type of activation for the last layer (None for no activation)
    bn: True/False, use batch-norm or not

    returns a Sequential module with layers of FC
    """

    # define activation
    activation = select_activation(activation)

    # build the cnn
    fc = []
    for i, (in_dim, out_dim) in enumerate(structure):
        fc.append(nn.Linear(in_dim, out_dim))
        # dont add activation to last layer unless stated otherwise
        if i == len(structure)-1 and last_activation:
            fc.append(select_activation(last_activation))
        elif activation and not (i == len(structure)-1):
            fc.append(activation)
        if bn:
            fc.append(nn.BatchNorm1d(out_dim))

    return nn.Sequential(*fc)


if __name__ == "__main__":
    print("fc layers (no last activation)")
    m = fc_layers(structure=[(100, 50), (50, 10)],
                  activation='relu',
                  bn=True)
    print(m)

    print("fc layers (last activation)")
    m = fc_layers(structure=[(100, 50), (50, 10)],
                  activation='relu',
                  bn=True,
                  last_activation='relu')
    print(m)

    print("cnn layers (no last activation)")
    m = cnn_layers(structure=[(3, 10, 3, 1, 1, 1), (10, 20, 3, 1, 1, 1)],
                   activation='relu',
                   pool=2,
                   bn=True)
    print(m)

    print("cnn layers (last activation)")
    m = cnn_layers(structure=[(3, 10, 3, 1, 1, 1), (10, 20, 3, 1, 1, 1)],
                   activation='tanh',
                   pool=4,
                   bn=False,
                   last_activation='sigmoid')
    print(m)
