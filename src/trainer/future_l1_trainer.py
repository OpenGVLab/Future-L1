import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class FutureL1SFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(FutureL1SFTTrainer, self).__init__(*args, **kwargs)
        # Buffer step-level metrics across gradient accumulation.
        # We flush (log) when `global_step` advances, so values are per optimizer step.
        self._loss_buf_step = None
        self._loss_buf_sum = {}
        self._loss_buf_cnt = 0

    def _accumulate_step_metrics(self, metrics: dict):
        """
        Accumulate per-microstep metrics and log averaged values once per optimizer step
        via HF Trainer's logging pipeline (e.g. W&B callback via `--report_to wandb`).
        """
        step = int(self.state.global_step or 0)

        # Initialize on first call.
        if self._loss_buf_step is None:
            self._loss_buf_step = step

        # If we've advanced to a new optimizer step, flush previous buffered metrics.
        if step != self._loss_buf_step:
            if self.is_world_process_zero() and self._loss_buf_cnt > 0:
                avg = {k: v / float(self._loss_buf_cnt) for k, v in self._loss_buf_sum.items()}
                # Keep key order stable for console/logger readability.
                ordered_avg = {}
                for k in ("train/ce_loss", "train/latent_loss"):
                    if k in avg:
                        ordered_avg[k] = avg[k]
                for k, v in avg.items():
                    if k not in ordered_avg:
                        ordered_avg[k] = v
                # Use Trainer.log so reporters (wandb/tensorboard/etc.) pick it up.
                self.log(ordered_avg)
            self._loss_buf_step = step
            self._loss_buf_sum = {}
            self._loss_buf_cnt = 0

        # Accumulate current microstep.
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if fv != fv:  # NaN
                continue
            self._loss_buf_sum[k] = self._loss_buf_sum.get(k, 0.0) + fv
        self._loss_buf_cnt += 1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss, accumulate for console logging (every logging_steps),
        and report auxiliary losses via HF Trainer logging (e.g. W&B).
        """
        outputs = model(**inputs)
        loss = outputs.loss

        ce_loss_val = None
        latent_loss_val = None
        recon_loss_val = None

        if hasattr(outputs, "ce_loss") and outputs.ce_loss is not None:
            try:
                ce_loss_val = outputs.ce_loss.detach().float().mean().item()
            except Exception:
                pass
        if hasattr(outputs, "latent_loss") and outputs.latent_loss is not None:
            try:
                latent_loss_val = outputs.latent_loss.detach().float().mean().item()
            except Exception:
                pass
        if hasattr(outputs, "recon_loss") and outputs.recon_loss is not None:
            try:
                recon_loss_val = outputs.recon_loss.detach().float().mean().item()
            except Exception:
                pass

        # Route metrics through HF Trainer logging (picked up by `--report_to wandb`).
        step_log = {}
        if ce_loss_val is not None:
            step_log["train/ce_loss"] = ce_loss_val
        if latent_loss_val is not None:
            step_log["train/latent_loss"] = latent_loss_val
        if recon_loss_val is not None:
            step_log["train/recon_loss"] = recon_loss_val
        if step_log:
            self._accumulate_step_metrics(step_log)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
                
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            save_strategy = getattr(self.args.save_strategy, "value", self.args.save_strategy)
            save_strategy = str(save_strategy).lower()
            if save_strategy in ("steps", "epoch") and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(FutureL1SFTTrainer, self)._save_checkpoint(model, trial)

        