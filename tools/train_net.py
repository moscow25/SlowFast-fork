#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import tempfile
import pandas as pd
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import model_builder
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter

logger = logging.get_logger(__name__)

# Simple loss function, removing discontinuity at 0-360 degrees
# Supports scaled inputs
def normal_degrees_loss(input, target, scale=1., offset=0., deg=360.):
    print(input, target, scale, deg)
    i_s, t_s = input*scale + offset, target*scale + offset
    i_s = torch.clamp(i_s, 0., deg)
    t_s = torch.clamp(t_s, 0., deg)
    a = i_s - t_s
    a = (a + 180.) % 360. - 180.
    a = torch.abs(a) / scale
    print(a)
    return torch.mean(a*a)

pi = 3.14159265358979323846
def deg2rad(t):
    return t * pi / 180.

def rad2deg(r):
    return r * 180. / pi

# Input will predict "tilt" as (x,y) vector; target is in degrees.
# A. Get both as vectors
# B. Cosine distance
def axis_tilt_similarity(input, target, use_tanh=False):
    #print(input, target)
    # Tanh -- saturates, and hard to get off the corners :-(
    # Without tanh -- make sure to penalize large values somehow...
    if use_tanh:
        input = torch.tanh(input)
    t = deg2rad(target).unsqueeze(1)
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    #print(cos_t, sin_t)
    t = torch.cat((cos_t, sin_t), dim=1)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #print(input, t)
    sim = cos(input, t)
    #print(sim)
    return sim

# For debugging, return degrees and clock from (x,y) as above
def tilt_from_xy(in_t, use_tanh=False):
    if use_tanh:
        in_t = torch.tanh(in_t)
    print(in_t)
    s = torch.sum((in_t * in_t), dim=1)
    print(s)
    in_t = in_t / s.unsqueeze(1)
    rad = torch.atan2(in_t[:,1], in_t[:,0])
    return rad2deg(rad)

# Unit norm -- penalize large values
def circle_unit_norm(in_t):
    s = torch.sum((in_t * in_t), dim=1)
    d = torch.ones_like(s)-s
    return torch.mean(d*d)

DEGREES_DIV_FACTOR = 90.
DEGREES_SUB_FACTOR = 2.
def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg,
    r_loss_weight=5.0, debug=False):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    # Hack -- save details and biggest errors
    outputs = []
    for cur_iter, (inputs, labels_tuple, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        # HACK -- return labels tuple -- categorical (classification), and regression values...
        labels = labels_tuple[0]
        labels = labels.cuda()
        labels_extra = labels_tuple[1]
        labels_extra = [l.unsqueeze(1) for l in labels_extra]
        labels_extra = torch.cat(labels_extra, dim=1)
        # HACK transform final column -- axis in degrees 0-360 --
        #labels_extra[:,-1] /= DEGREES_DIV_FACTOR
        #labels_extra[:,-1] -= DEGREES_SUB_FACTOR
        names = labels_tuple[2]

        # saving data for saving validation
        if debug:
            names = np.concatenate((np.expand_dims(labels_tuple[2][0], axis=1),
                np.expand_dims(labels_tuple[2][1], axis=1),
                np.expand_dims(labels_tuple[2][2].numpy(), axis=1)), axis=1)
            print(names.shape, np.expand_dims(labels_tuple[0].numpy(), axis=1).shape,
                labels_tuple[1][0].numpy().shape, labels_tuple[1][1].numpy().shape)
            val_outs = np.concatenate((names, np.expand_dims(labels_tuple[0].numpy(), axis=1),
                labels_extra.numpy()), axis=1)
            print(val_outs)

        if debug:
            print(labels)
            print(labels_extra)
            print(labels_extra.shape)
        labels_extra = labels_extra.cuda().float()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds_all = model(inputs, meta["boxes"])

        else:
            # Perform the forward pass.
            preds_all = model(inputs)

        # HACK -- Output multiple predictions.
        if True:
            preds, preds_extra = preds_all
        else:
            preds = preds_all
            preds_extra = None

        if debug:
            print(preds)
            print(preds_extra)
            print(tilt_from_xy(preds_extra[:,-2:]))
            print(names)

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction='none')#(reduction="mean")
        # Apply "Huber loss" to regression terms -- L2 under norm 1.0, L1 otherwise (not as sensitive to outliers)
        #loss_fun_extra = torch.nn.SmoothL1Loss()
        loss_fun_extra = torch.nn.MSELoss(reduction='none')
        loss_fun_degrees = normal_degrees_loss

        # Compute the loss.
        c_loss = loss_fun(preds, labels)
        #r_loss = loss_fun_extra(preds_extra, labels_extra)
        # HACK -- "normal" regression loss, except the last two columns ([x,y] for spin axis)
        r_loss = loss_fun_extra(preds_extra[:,:-2], labels_extra[:,:-1])
        w_loss = torch.mean((c_loss.unsqueeze(1) + r_loss*r_loss_weight)/(1.0+r_loss_weight), dim=1)

        if debug:
            print(r_loss)
            print(c_loss)
            print('losses')
            print(w_loss)
            #print(val_outs.shape, preds.shape, preds_extra.shape, w_loss.shape)
            val_outs = np.concatenate((val_outs, preds.detach().cpu().numpy(),
                preds_extra.detach().cpu().numpy(),
                c_loss.unsqueeze(1).detach().cpu().numpy(),
                torch.mean(r_loss, dim=1).unsqueeze(1).detach().cpu().numpy(),
                w_loss.unsqueeze(1).detach().cpu().numpy()), axis=1)
            outputs.append(val_outs)
        # Manual reduce
        c_loss = torch.mean(c_loss)
        r_loss = torch.mean(r_loss)

        axis_loss = (1.0 - axis_tilt_similarity(preds_extra[:,-2:], labels_extra[:,-1]))
        # TODO -- Square the cosine similarity loss for axis?
        # TODO -- in debug mode, display degrees from x,y pedictions... harder to read otherwise.
        axis_loss = torch.mean(axis_loss)
        r_loss = (r_loss * (preds_extra.shape[1]-2) + axis_loss*2)/preds_extra.shape[1]

        # HACK -- a kind of "weight norm" on coordinates prediction
        w_loss = circle_unit_norm(preds_extra[:,-2:]) * 0.001

        #r_loss += loss_fun_degrees(preds_extra[:,-1], labels_extra[:,-1],
        #    scale=DEGREES_DIV_FACTOR, offset=DEGREES_SUB_FACTOR*DEGREES_DIV_FACTOR)
        # Add weighting parameter to loss parts. (Ignore or over-weight regression loss r_loss)
        r_weight = r_loss_weight
        c_weight = 1.0
        loss = (c_weight*c_loss + r_weight*r_loss) / (r_weight + c_weight) + w_loss

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 2))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

            # Count the errors, for regression [normalized, so w/r/t stdev]
            # HACK -- average across all predictions, in all categories
            reg_err_05 = metrics.regression_correct(preds_extra[:,:-2], labels_extra[:,:-1], eps=0.5) / preds_extra.shape[1]
            reg_err_025 = metrics.regression_correct(preds_extra[:,:-2], labels_extra[:,:-1], eps=0.25) / preds_extra.shape[1]
            if debug:
                print('0.5 and 0.25 regression error rates:')
                print(reg_err_05, reg_err_025)

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, c_loss, r_loss, top1_err, top5_err, reg_err_05, reg_err_025 = du.all_reduce(
                    [loss, c_loss, r_loss, top1_err, top5_err, reg_err_05, reg_err_025]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss, c_loss, r_loss, top1_err, top5_err, reg_err_05, reg_err_025 = (
                loss.item(),
                c_loss.item(),
                r_loss.item(),
                top1_err.item(),
                top5_err.item(),
                reg_err_05.item(),
                reg_err_025.item(),
            )

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                top1_err, top5_err, loss, c_loss, r_loss, lr, inputs[0].size(0) * cfg.NUM_GPUS, reg_err_05, reg_err_025
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    if debug:
        outputs = np.concatenate(outputs, axis=0)
        def last(n): return n[-1]
        outputs = outputs[outputs[:,-1].argsort()]
        print(outputs)
        print(outputs.shape)

        for k in outputs[-10:, :].tolist():
            print(k)

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    #assert False


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, r_loss_weight=5., debug=False):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    # Hack -- save details and biggest errors
    outputs = []
    output_header = ['filepath', 'spinAxis', 'spinAxisDeg', 'pitchType',
        'speed_norm', 'spin_norm', 'trueSpin_norm',
        'spinEfficiency_norm', 'topSpin_norm', 'sideSpin_norm',
        'rifleSpin_norm', 'vb_norm', 'hb_norm', 'hAngle_norm', 'rAngle_norm',
        'spinAxisDeg_copy',
        'pitchType0_logit', 'pitchType1_logit', 'pitchType2_logit', 'pitchType3_logit',
        'pitchType4_logit', 'pitchType5_logit', 'pitchType6_logit', 'pitchType7_logit',
        'speed_norm_pred', 'spin_norm_pred', 'trueSpin_norm_pred',
        'spinEfficiency_norm_pred', 'topSpin_norm_pred', 'sideSpin_norm_pred',
        'rifleSpin_norm_pred', 'vb_norm_pred', 'hb_norm_pred', 'hAngle_norm_pred', 'rAngle_norm_pred',
        'spinAxis_X_pred', 'spinAxis_Y_pred',
        'c_loss', 'r_loss', 'total_loss']

    for cur_iter, (inputs, labels_tuple, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        # HACK -- return labels tuple -- categorical (classification), and regression values...
        labels = labels_tuple[0]
        labels = labels.cuda()
        labels_extra = labels_tuple[1]
        labels_extra = [l.unsqueeze(1) for l in labels_extra]
        labels_extra = torch.cat(labels_extra, dim=1)
        names = labels_tuple[2]

        # saving data for saving validation
        if debug:
            names = np.concatenate((np.expand_dims(labels_tuple[2][0], axis=1),
                np.expand_dims(labels_tuple[2][1], axis=1),
                np.expand_dims(labels_tuple[2][2].numpy(), axis=1)), axis=1)
            #print(names.shape, np.expand_dims(labels_tuple[0].numpy(), axis=1).shape,
            #    labels_tuple[1][0].numpy().shape, labels_tuple[1][1].numpy().shape)
            val_outs = np.concatenate((names, np.expand_dims(labels_tuple[0].numpy(), axis=1),
                labels_extra.numpy()), axis=1)
            #print(val_outs)

        #if debug:
        #    print(labels)
        #    print(labels_extra)
        #    print(labels_extra.shape)
        labels_extra = labels_extra.cuda().float()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
        else:
            preds_all = model(inputs)
            # HACK -- Output multiple predictions.
            if True:
                preds, preds_extra = preds_all
            else:
                preds = preds_all
                preds_extra = None

            #if debug:
            #    print(preds, preds_extra)
            #    print(names)

            # Save validation losses.
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="none")#(reduction="mean")
            # Apply "Huber loss" to regression terms -- L2 under norm 1.0, L1 otherwise (not as sensitive to outliers)
            #loss_fun_extra = torch.nn.SmoothL1Loss()
            loss_fun_extra = torch.nn.MSELoss(reduction="none")
            c_loss = loss_fun(preds, labels)
            #r_loss = loss_fun_extra(preds_extra, labels_extra)
            # HACK -- "normal" regression loss, except the last two columns ([x,y] for spin axis)
            r_loss = loss_fun_extra(preds_extra[:,:-2], labels_extra[:,:-1])
            w_loss = torch.mean((c_loss.unsqueeze(1) + r_loss*r_loss_weight)/(1.0+r_loss_weight), dim=1)

            if debug:
                #print(r_loss)
                #print(c_loss)

                #print('losses')
                #print(w_loss)
                #print(val_outs.shape, preds.shape, preds_extra.shape, w_loss.shape)
                val_outs = np.concatenate((val_outs, preds.detach().cpu().numpy(),
                    preds_extra.detach().cpu().numpy(),
                    c_loss.unsqueeze(1).detach().cpu().numpy(),
                    torch.mean(r_loss, dim=1).unsqueeze(1).detach().cpu().numpy(),
                    w_loss.unsqueeze(1).detach().cpu().numpy()), axis=1)
                outputs.append(val_outs)
            # Reduce manually
            c_loss = torch.mean(c_loss)
            r_loss = torch.mean(r_loss)

            axis_loss = (1.0 - axis_tilt_similarity(preds_extra[:,-2:], labels_extra[:,-1]))
            axis_loss = torch.mean(axis_loss)
            r_loss = (r_loss * (preds_extra.shape[1]-2) + axis_loss*2)/preds_extra.shape[1]

            # Add weighting parameter to loss parts. (Ignore or over-weight regression loss r_loss)
            r_weight = r_loss_weight
            c_weight = 1.0
            loss = (c_weight*c_loss + r_weight*r_loss) / (r_weight + c_weight)

            # check Nan Loss.
            misc.check_nan_losses(loss)

            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 2))

            # Count the errors, for regression [normalized, so w/r/t stdev]
            # HACK -- average across all predictions, in all categories
            reg_err_05 = metrics.regression_correct(preds_extra[:,:-2], labels_extra[:,:-1], eps=0.5) / preds_extra.shape[1]
            reg_err_025 = metrics.regression_correct(preds_extra[:,:-2], labels_extra[:,:-1], eps=0.25) / preds_extra.shape[1]

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err, loss, c_loss, r_loss, reg_err_05, reg_err_025 = du.all_reduce([top1_err, top5_err,
                    loss, c_loss, r_loss, reg_err_05, reg_err_025])


            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err, loss, c_loss, r_loss = top1_err.item(), top5_err.item(), loss.item(), c_loss.item(), r_loss.item()
            reg_err_05, reg_err_025 = reg_err_05.item(), reg_err_025.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err, top5_err, loss, c_loss, r_loss, 0., inputs[0].size(0) * cfg.NUM_GPUS, reg_err_05, reg_err_025
            )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    if debug:
        outputs = np.concatenate(outputs, axis=0)
        def last(n): return n[-1]
        outputs = outputs[outputs[:,-1].argsort()]
        print(outputs)
        print(outputs.shape)
        for k in outputs[-20:, :].tolist():
            print(k)

        # Save data to local path.
        df = pd.DataFrame(data=outputs, columns=output_header)
        val_df_path = './val_epoch_%s_%s.csv' % (cur_epoch, next(tempfile._get_candidate_names()))
        print('Saving %s val examples to %s' % (str(df.shape), val_df_path))
        df.to_csv(val_df_path, index=False)

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        #val_meter = ValMeter(len(val_loader), cfg)
        # Too lazy to add other values...
        val_meter = TrainMeter(len(val_loader), cfg, is_val=True)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
