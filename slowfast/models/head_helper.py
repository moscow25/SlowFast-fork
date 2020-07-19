#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from detectron2.layers import ROIAlign
from itertools import chain


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        mlp_sizes = [],
        softmax_outputs = [],
        mlp_dropout = 0.1,
        mlp_norm = 'layer',
        use_maxpool = False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        # Hack! Try adding max pools as well?
        self.use_maxpool = use_maxpool
        print('Using pool size/shape:')
        print(pool_size)
        if self.use_maxpool:
            dim_in *= 2
        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

            if self.use_maxpool:
                max_pool = nn.MaxPool3d(pool_size[pathway], stride=1)
                self.add_module("pathway{}_maxpool".format(pathway), max_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        #init_proj =

        # normalization?
        # TODO -- set as optional parameter
        if mlp_norm and mlp_norm.lower() == 'layer':
            self.norm_layer = nn.LayerNorm(sum(dim_in), elementwise_affine=False)
        else:
            self.norm_layer = nn.Identity()

        self.layer_sizes = [sum(dim_in)] + list(map(int, mlp_sizes)) + [num_classes]
        self.nonlinearity = nn.LeakyReLU(0.2)
        self.has_final_layers = False
        if len(self.layer_sizes) <= 2:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        else:
            layer_list = []
            #layer_list.extend([nn.Dropout(dropout_rate)])

            layer_list.extend(list(chain.from_iterable(
                [[nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]), self.nonlinearity,  nn.Dropout(mlp_dropout), ] for i in range(len(self.layer_sizes) - 1)]
            )))
            # HACK -- drop nonlinearity and dropout, for final output.
            if len(softmax_outputs) == 0:
                layer_list = layer_list[:-2]
                #layer_list = layer_list[:-3]
                print('Projection layer:')
                print(layer_list)
                self.projection = nn.Sequential(*layer_list)
                self.has_final_layers = False
            else:
                # HACK -- separate out at final layer... produce several outputs
                layer_list = layer_list[:-3]
                #layer_list = layer_list[:-4]
                print('Projection layer:')
                print(layer_list)
                self.projection = nn.Sequential(*layer_list)

                # create multiple final layers...
                final_layers = [num_classes] + softmax_outputs
                fl = []
                for s in final_layers:
                    l = nn.Linear(self.layer_sizes[-2],s)
                    fl.append(l)
                self.has_final_layers = True
                self.final_layers = nn.ModuleList(fl)

        # Softmax for evaluation and testing.
        # activation for *final* outputs [non for regression!]
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "tanh":
            self.act == nn.Tanh()
        elif act_func == "none":
            print('applying none/Identity activation for head')
            self.act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, debug=False):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        if debug:
            print('Debug HEAD forward pass')
            print('examining %d pathways' % self.num_pathways)
            print('Inputs:')
            print([i.shape for i in inputs])
            print('~')
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            t = m(inputs[pathway])
            pool_out.append(t)

            if debug:
                print(t.shape)

            if self.use_maxpool:
                m = getattr(self, "pathway{}_maxpool".format(pathway))
                pool_out.append(m(inputs[pathway]))

        x = torch.cat(pool_out, 1)

        if debug:
            print('cat all pools output')
            print(len(pool_out))
            print(x.shape)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        # Layer Norm *before* dropout
        if hasattr(self, "norm_layer"):
            x = self.norm_layer(x)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if self.has_final_layers:
            xl = []
            for l in self.final_layers:
                xf = l(x)
                if debug:
                    print(xf.shape)
                xf = xf.view(xf.shape[0], -1)
                xl.append(xf)
        else:
            x = x.view(x.shape[0], -1)
            xl = x

        # Performs fully convolutional inference.
        # TODO: Why is this always called? Confusing.
        # TODO -- should only be used in *test* not validation. Report bug.
        """
        if not self.training:
            # HACK: Try actually applying the activation function? Will be softmax, most likely.
            x = self.act(x)
            x = x.mean([1, 2, 3])
        """
        if debug:
            print('returning XL')
            print(len(xl), [l.shape for l in xl])

        #x = x.view(x.shape[0], -1)
        return xl
