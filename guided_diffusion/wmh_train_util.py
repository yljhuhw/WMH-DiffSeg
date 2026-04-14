"""
WMHTrainLoop: inherits TrainLoop, overrides forward_backward to use
training_losses_segmentation_wmh.
"""
import functools
import torch as th
from .train_util import TrainLoop, log_loss_dict
from . import dist_util, logger
from .resample import LossAwareSampler


class WMHTrainLoop(TrainLoop):
    def __init__(self, *args, lambda_dice=0.1, lambda_cross_scale=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_dice = lambda_dice
        self.lambda_cross_scale = lambda_cross_scale

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation_wmh,
                self.ddp_model,
                micro,
                t,
                lambda_dice=self.lambda_dice,
                lambda_cross_scale=self.lambda_cross_scale,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )

            losses = losses1[0]
            sample = losses1[1]
            loss = (losses["loss"] * weights + losses["loss_cal"] * 10).mean()

            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)
        return sample
