import torch
import torch.nn as nn
import os
from tools import builder
from utils import misc, dist_utils
import time
import wandb
from utils.logger import print_log, get_logger
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from datasets import data_transforms
from utils.gaussian import write_gaussian_feature_to_ply, unnormalize_gaussians

train_transforms = data_transforms.PointcloudScaleAndTranslate()


class Acc_Metric:
    def __init__(self, acc=0.0):
        if type(acc).__name__ == "dict":
            self.acc = acc["acc"]
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict["acc"] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1.0 / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (
        (train_sampler, train_dataloader),
        (_, test_dataloader),
    ) = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(
        args, config.dataset.val
    )
    (_, extra_train_dataloader) = (
        builder.dataset_builder(args, config.dataset.extra_train)
        if config.dataset.get("extra_train")
        else (None, None)
    )
    # build model
    # pass the norm attribute to the model
    config.model.norm_attribute = config.dataset.train.others.norm_attribute
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        print("Using GPU of Rank", args.local_rank)
        base_model = base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.0)
    metrics = Acc_Metric(0.0)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(
            base_model, args, logger=logger, strict_load=True
        )
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(
            base_model, args.start_ckpts, logger=logger, strict_load=True
        )

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # training
    # CD loss is ~1e-3 scale so historically multiplied by 1000 for readability.
    # JEPA latent loss is already human-readable — set loss_display_scale: 1 in config.
    loss_scale = getattr(config, "loss_display_scale", 1000)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):

        if epoch == config.max_epoch:
            final_recon_dict = {}  # keys populated dynamically from first batch's loss_dict

        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(["Loss"])

        num_iter = 0

        base_model.train()  # set model to training mode
        base_model.zero_grad()
        n_batches = len(train_dataloader)
        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data, scale_c, scale_m) in enumerate(
            train_dataloader
        ):

            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME
            points = data.cuda()

            if config.npoints_fps:
                # using fps gs to select subset of points
                points = misc.fps_gs(
                    points, npoints, attribute=config.model.group_attribute
                )
            else:
                # using random sampling
                random_idx = np.random.choice(points.size(1), npoints, False)
                points = points[:, random_idx, :].contiguous()

            model_unwrapped = (
                base_model.module if hasattr(base_model, "module") else base_model
            )
            supports_ply_save = getattr(
                model_unwrapped,
                "supports_ply_save",
                hasattr(model_unwrapped, "increase_dim"),
            )

            if (
                epoch == config.max_epoch and idx % 5000 == 0 and supports_ply_save
            ):  # save last epoch ply for visualization (MAE only)
                loss_dict, vis_gaussians, full_rebuild_gaussian, original_gaussians = (
                    base_model(points, save=True)
                )
                # save to gaussian ply
                os.makedirs(
                    os.path.join(args.experiment_path, "save_ply"), exist_ok=True
                )

                original_gaussians, vis_gaussians, full_rebuild_gaussian = (
                    unnormalize_gaussians(
                        original_gaussians,
                        vis_gaussians,
                        full_rebuild_gaussian,
                        scale_c,
                        scale_m,
                        config,
                    )
                )

                for i in range(vis_gaussians.shape[0]):  # save whole batch
                    vis_gaussians_ply_path = os.path.join(
                        args.experiment_path,
                        "save_ply",
                        f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_vis_gaussians.ply",
                    )
                    full_rebuild_gaussian_ply_path = os.path.join(
                        args.experiment_path,
                        "save_ply",
                        f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_full_rebuild_gaussian.ply",
                    )
                    original_gaussians_ply_path = os.path.join(
                        args.experiment_path,
                        "save_ply",
                        f"{model_ids[i]}_original_gaussians.ply",
                    )
                    write_gaussian_feature_to_ply(
                        vis_gaussians[i], vis_gaussians_ply_path
                    )
                    write_gaussian_feature_to_ply(
                        full_rebuild_gaussian[i], full_rebuild_gaussian_ply_path
                    )
                    write_gaussian_feature_to_ply(
                        original_gaussians[i], original_gaussians_ply_path
                    )
            else:
                if epoch != config.max_epoch:
                    points = train_transforms(points)
                loss_dict = base_model(points)

            # aggregate all loss
            loss = sum([loss_dict[key] for key in loss_dict.keys()])
            try:
                loss.backward()
                # Using one gpu
            except:
                loss = loss.mean()
                loss.backward()
                # "Using multi GPUs"

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

                # EMA teacher update (only active for models that implement _update_teacher)
                model_unwrapped = base_model.module if hasattr(base_model, "module") else base_model
                if hasattr(model_unwrapped, "_update_teacher"):
                    total_steps = n_batches * config.max_epoch
                    ema_start = getattr(config, "ema_momentum_start", 0.996)
                    ema_end = getattr(config, "ema_momentum_end", 0.9999)
                    ema_m = ema_start + (ema_end - ema_start) * n_itr / max(total_steps, 1)
                    model_unwrapped._update_teacher(momentum=ema_m)

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * loss_scale])

            else:
                losses.update([loss.mean().item() * loss_scale])
                loss_dict = {
                    key: loss_dict[key].mean().item() * loss_scale for key in loss_dict.keys()
                }

                if epoch == config.max_epoch:
                    for key in loss_dict.keys():
                        if key not in final_recon_dict:
                            final_recon_dict[key] = []
                        final_recon_dict[key].append(loss_dict[key])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar("Loss/Batch/Loss", loss.item(), n_itr)
                # use loss dict to add scaler
                for key in loss_dict.keys():
                    train_writer.add_scalar(f"Loss/Batch/{key}", loss_dict[key], n_itr)
                train_writer.add_scalar(
                    "Loss/Batch/LR", optimizer.param_groups[0]["lr"], n_itr
                )

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:

                print_log(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f"
                    % (
                        epoch,
                        config.max_epoch,
                        idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ["%.4f" % l for l in losses.val()],
                        optimizer.param_groups[0]["lr"],
                    ),
                    logger=logger,
                )
                # print each loss at true scale (loss_dict already scaled, divide back)
                for key in loss_dict.keys():
                    print_log(f"{key} = {loss_dict[key] / loss_scale}", logger=logger)
                # print all kind of loss
                if args.use_wandb:
                    for key in loss_dict.keys():
                        wandb.log({key: loss_dict[key]}, step=n_itr)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar("Loss/Epoch/Loss_1", losses.avg(0), epoch)
        print_log(
            "[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f"
            % (
                epoch,
                epoch_end_time - epoch_start_time,
                ["%.4f" % l for l in losses.avg()],
                optimizer.param_groups[0]["lr"],
            ),
            logger=logger,
        )

        builder.save_checkpoint(
            base_model,
            optimizer,
            epoch,
            metrics,
            best_metrics,
            "ckpt-last",
            args,
            logger=logger,
        )
        if (epoch < 200 and epoch % 50 == 0) or (epoch >= 200 and epoch % 25 == 0) or epoch == config.max_epoch:
            builder.save_checkpoint(
                base_model,
                optimizer,
                epoch,
                metrics,
                best_metrics,
                f"ckpt-epoch-{epoch:03d}",
                args,
                logger=logger,
            )

    for attribute, value in final_recon_dict.items():
        print(f"{attribute} loss: {np.mean(value)}")

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(
    base_model,
    extra_train_dataloader,
    test_dataloader,
    epoch,
    val_writer,
    args,
    config,
    logger=None,
):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.N_POINTS
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(
            train_features.data.cpu().numpy(),
            train_label.data.cpu().numpy(),
            test_features.data.cpu().numpy(),
            test_label.data.cpu().numpy(),
        )

        print_log(
            "[Validation] EPOCH: %d  acc = %.4f" % (epoch, svm_acc), logger=logger
        )

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar("Metric/ACC", svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
