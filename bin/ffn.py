import datetime
import math
import statistics
import typing
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, Protocol, TypedDict

import delu
import numpy as np
import rtdl_num_embeddings
import rtdl_revisiting_models
import scipy.special
import tabm
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

import lib
import lib.data
import lib.deep
import lib.env
import lib.experiment
import lib.optim
import lib.util


from lib.types import KWArgs, PartKey, PredictionType


def make_num_embeddings(type: str, **kwargs) -> nn.Module:
    return getattr(rtdl_num_embeddings, type)(**kwargs)


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        type: Literal['mlp', 'tabm'],
        num_embeddings: None | KWArgs = None,
        num_embeddings_bins: None | list[Tensor],
        backbone: KWArgs,
    ) -> None:
        assert n_num_features > 0 or cat_cardinalities

        super().__init__()

        d_features: list[int] = []

        if num_embeddings is None:
            self.num_module = None
            d_one_num_feature = 1
        else:
            assert n_num_features > 0
            self.num_module = make_num_embeddings(
                **num_embeddings,  # type: ignore
                **(
                    {'n_features': n_num_features}
                    if num_embeddings_bins is None
                    else {'bins': num_embeddings_bins}
                ),
            )
            d_one_num_feature = num_embeddings['d_embedding']
        d_in_num = d_one_num_feature * n_num_features
        d_features.extend(d_one_num_feature for _ in range(n_num_features))

        self.cat_module = (
            lib.deep.OneHotEncoding(cat_cardinalities) if cat_cardinalities else None
        )
        d_in_cat = sum(cat_cardinalities)
        d_features.extend(cat_cardinalities)

        d_in = d_in_num + d_in_cat
        d_out = 1 if n_classes is None or n_classes == 2 else n_classes

        if type == 'mlp':
            self.ensemble_view = None
            self.backbone = tabm.MLPBackbone(d_in=d_in, **backbone)
            self.output = nn.Linear(backbone['d_block'], d_out)

        elif type == 'tabm':
            is_tabm_packed = backbone.get('arch_type', 'tabm') == 'tabm-packed'
            self.ensemble_view = tabm.EnsembleView(k=backbone['k'])
            self.backbone = tabm.make_tabm_backbone(
                d_in=d_in,
                **backbone,
                start_scaling_init=None if is_tabm_packed else 'random-signs',
                start_scaling_init_chunks=None if is_tabm_packed else d_features,
            )
            self.output = tabm.LinearEnsemble(
                self.backbone.get_original_output_shape()[0], d_out, k=self.backbone.k
            )

        else:
            raise ValueError(f'Unknown {type=}')

        self._n_num_features = n_num_features

    def forward(self, x_num: None | Tensor, x_cat: None | Tensor) -> Tensor:
        x_list: list[Tensor] = []

        if x_num is None:
            assert self._n_num_features == 0
        else:
            assert self._n_num_features > 0
            if self.num_module is None:
                x_list.append(x_num)
            elif x_num.ndim == 2:  # (B, D_num)
                x_list.append(self.num_module(x_num).flatten(-2))
            else:
                assert x_num.ndim == 3  # (B, K, D_num)
                # Some embedding modules support only the (B, D) input shape.
                x_list.append(
                    self.num_module(x_num.flatten(0, 1))
                    .unflatten(0, x_num.shape[:2])
                    .flatten(-2)
                )

        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x_list.append(self.cat_module(x_cat).to(torch.get_default_dtype()))

        x = torch.cat(x_list, dim=-1)

        if self.ensemble_view is not None:
            x = self.ensemble_view(x)
        x = self.backbone(x)
        x = self.output(x)
        return x


_CUSTOM_OPTIMIZERS = {
    x.__name__: x
    for x in [
        lib.optim.Adan,
        lib.optim.AdEMAMix,
        lib.optim.ADOPT,
        lib.optim.AdaBelief,
        lib.optim.Lion,
        lib.optim.Signum,
        lib.optim.SOAP,
        lib.optim.AdamWScheduleFree,
    ]
}
_CUSTOM_OPTIMIZERS['C-AdamW'] = lib.optim.c_adamw.AdamW
_CUSTOM_OPTIMIZERS['Muon'] = lib.optim.muon.SingleDeviceMuonWithAuxAdam


def get_optimizer_class(type: str) -> type[torch.optim.Optimizer]:
    Optimizer = getattr(torch.optim, type, None)
    if Optimizer is None:
        Optimizer = _CUSTOM_OPTIMIZERS[type]
    return Optimizer


def make_optimizer(
    model: Model,
    type: str,
    muon: dict[str, Any] | None = None,
    **optimizer_config_kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer and split the model parameters into PyTorch parameter groups.

    PyTorch optimizers can be given either a flat iterable of parameters or a
    list of *parameter groups*. A parameter group is a dict with a mandatory
    `"params"` key and optional optimizer hyperparameters (`lr`, `weight_decay`,
    etc.) that override the defaults for that subset of parameters. This is the
    standard PyTorch way to apply different optimization rules to different
    parts of the same model.

    This function splits parameters into disjoint groups and sets each group
    hypeprparameters explicitly.

    The parameter allocation policy in this function is the following:

    1. *Muon group* (optional). This group is non-empty when `type` is `"Muon"` or
    `"Muon-PolarExpress"`). Muon is assigned only to matrix-like weights of the
    backbone. It is not applied to the output layer, or the numerical embedding
    layers. The Muon group inherits the common `lr` and `weight_decay` and
    overrides them with the `muon` dict provided. When Muon is enabled, the Muon
    group is marked with `use_muon=True`, while the remaining groups are marked
    with `use_muon=False`

    2. *Zero-weight-decay group*. Biases, normalization parameters, and
    numerical-embedding parameters are placed into a separate group with
    `weight_decay=0.0`.

    3. *Default group*. All remaining parameters are placed into the default
    group and use the usual optimizer hyperparameters from
    `optimizer_config_kwargs`.
    """

    optimizer_cls = get_optimizer_class(type)
    param_groups = []

    if type != 'Muon':
        nonmuon_hparams = {
            **optimizer_config_kwargs,
        }
        muon_params = []
    else:
        nonmuon_hparams = {
            'use_muon': False,
            **optimizer_config_kwargs,
        }

        # the latter `**` overrides the common hyperparameters, thus `muon` dict
        # allows to specify muon-specify hyperparameters
        muon_hparams = {
            **{x: optimizer_config_kwargs[x] for x in ['lr', 'weight_decay']},
            **({} if muon is None else muon),
        }
        muon_params = frozenset(
            [
                typing.cast(nn.Parameter, x.weight)
                for x in model.backbone.modules()
                if isinstance(
                    x, nn.Linear | tabm.LinearBatchEnsemble | tabm.LinearEnsemble
                )
                and 1 not in x.weight.shape[-2:]
            ]
        )

        muon_group = {
            'params': muon_params,
            'use_muon': True,
            **muon_hparams,
        }
        param_groups.append(muon_group)

    zero_wd_params = frozenset(
        [
            p
            for m in model.modules()
            for pn, p in m.named_parameters()
            if (
                isinstance(
                    m,
                    nn.BatchNorm1d
                    | nn.LayerNorm
                    | nn.InstanceNorm1d
                    | rtdl_revisiting_models.LinearEmbeddings
                    | rtdl_num_embeddings.LinearEmbeddings
                    | rtdl_num_embeddings.LinearReLUEmbeddings
                    | rtdl_num_embeddings._Periodic,
                )
                or pn.endswith('bias')
            )
        ]
    )
    zero_wd_group = {
        'params': list(zero_wd_params),
        **nonmuon_hparams,
        'weight_decay': 0.0,
    }
    param_groups.append(zero_wd_group)

    default_group = {
        'params': [
            p
            for p in model.parameters()
            if p not in muon_params and p not in zero_wd_params
        ],
        **nonmuon_hparams,
    }
    param_groups.append(default_group)

    return optimizer_cls(param_groups)  # type: ignore


class ApplyModel(Protocol):
    def __call__(
        self,
        model: nn.Module,
        dataset: lib.data.Dataset,
        *,
        part: PartKey,
        idx: Tensor,
    ) -> Tensor: ...


def apply_model_impl(
    model: nn.Module, dataset: lib.data.Dataset, *, part: PartKey, idx: Tensor
) -> Tensor:
    return (
        model(
            dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None,
            dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression predictions.
        .float()
    )


class EvaluateImplOutput(NamedTuple):
    metrics: dict[PartKey, Any]
    predictions: dict[PartKey, np.ndarray]


@lib.util.adjust_gpu_memory_usage('batch_size')
def evaluate_impl(
    apply_model: ApplyModel,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    is_tabm: bool,
    dataset: lib.data.Dataset,
    *,
    parts: list[PartKey],
    regression_label_stats: None | lib.data.RegressionLabelStats,
    prediction_type: str | PredictionType,
    batch_size: int,
    device: torch.device,
) -> EvaluateImplOutput:
    model.eval()
    if isinstance(optimizer, lib.optim.AdamWScheduleFree):
        optimizer.eval()

    predictions = {
        part: (
            torch.cat(
                [
                    apply_model(model, dataset, part=part, idx=idx)
                    for idx in torch.arange(dataset.size(part), device=device).split(
                        batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        for part in parts
    }

    if dataset.task.is_regression:
        assert regression_label_stats is not None
        for part in predictions:
            predictions[part] *= regression_label_stats.std
            predictions[part] += regression_label_stats.mean
    elif dataset.task.is_binclass:
        predictions = {k: scipy.special.expit(v) for k, v in predictions.items()}
    else:
        assert dataset.task.is_multiclass
        predictions = {
            k: scipy.special.softmax(v, axis=-1) for k, v in predictions.items()
        }

    if is_tabm:
        predictions = {k: v.mean(1) for k, v in predictions.items()}

    metrics = (
        dataset.task.calculate_metrics(predictions, prediction_type)
        if lib.util.are_valid_predictions(predictions)
        else {x: {'score': lib.util.WORST_SCORE} for x in predictions}
    )

    return EvaluateImplOutput(metrics, predictions)


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    model_num_embeddings_bins: NotRequired[KWArgs]
    optimizer: KWArgs
    batch_size: int
    eval_batch_size: NotRequired[int]
    share_training_batches: NotRequired[bool]
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    save_checkpoint: NotRequired[bool]


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])
    device = lib.util.get_device()
    logger.info(f'Device: {device}')

    # >>> Data
    dataset = lib.data.build_dataset(**config['data'])
    assert dataset.n_bin_features == 0
    regression_label_stats = dataset.try_standardize_labels_()
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_multiclass else torch.float
    )

    # >>> Model
    if 'model_num_embeddings_bins' in config:
        compute_bins_kwargs = (
            {
                'y': Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                'regression': dataset.task.is_regression,
                'verbose': True,
            }
            if 'tree_kwargs' in config['model_num_embeddings_bins']
            else {}
        )
        model_num_embeddings_bins = rtdl_num_embeddings.compute_bins(
            dataset.data['x_num']['train'],
            **config['model_num_embeddings_bins'],
            **compute_bins_kwargs,
        )
        logger.info(f'Bin counts: {[len(x) - 1 for x in model_num_embeddings_bins]}')
    else:
        model_num_embeddings_bins = None
    is_tabm = config['model']['type'] == 'tabm'
    model: Model = Model(
        n_num_features=dataset.n_num_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=dataset.task.try_compute_n_classes(),
        num_embeddings_bins=model_num_embeddings_bins,
        **config['model'],
    )
    tabm_k = (
        typing.cast(tabm.MLPBackboneEnsemble, model.backbone).k if is_tabm else None
    )

    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters: {report["n_parameters"]}')
    report['prediction_type'] = prediction_type = (
        'labels' if dataset.task.is_regression else 'probs'
    )
    model.to(device)

    # >>> Training
    ema_decay = config['optimizer'].pop('ema_decay', None)
    ema_model = (
        AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
            use_buffers=True,
        )
        if ema_decay is not None
        else None
    )

    optimizer = make_optimizer(model, **config['optimizer'])

    gradient_clipping_norm = config.get('gradient_clipping_norm')
    base_loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.binary_cross_entropy_with_logits
        if dataset.task.is_binclass
        else nn.functional.cross_entropy
    )

    if is_tabm:
        assert tabm_k is not None

        def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
            return base_loss_fn(
                y_pred.flatten(0, 1),
                (
                    y_true.repeat_interleave(tabm_k)
                    if share_training_batches
                    else y_true.flatten(0, 1)
                ),
            )

    else:
        loss_fn = base_loss_fn  # type: ignore

    step = 0
    train_size = dataset.size('train')
    batch_size = config['batch_size']
    report['epoch_size'] = epoch_size = math.ceil(dataset.size('train') / batch_size)
    eval_batch_size = config.get('eval_batch_size', 32768)

    share_training_batches = config.get('share_training_batches')
    if is_tabm:
        assert share_training_batches is not None
    else:
        assert share_training_batches is None
        share_training_batches = True

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    timer = delu.tools.Timer()
    best_checkpoint = None
    writer = torch.utils.tensorboard.SummaryWriter(exp)

    # >>> Functions
    apply_model = apply_model_impl
    eval_model = ema_model if ema_model is not None else model
    # The following order of `evaluation_mode` and `partial` preserves
    # typing-related hints in VSCode.
    evaluate = torch.inference_mode()(
        partial(
            evaluate_impl,
            apply_model,
            eval_model,
            optimizer,
            is_tabm,
            dataset,
            regression_label_stats=regression_label_stats,
            prediction_type=prediction_type,
            device=device,
        )
    )

    def make_checkpoint() -> dict[str, Any]:
        checkpoint = {
            'step': step,
            'model': model.state_dict(),
            **({} if ema_model is None else {'ema_model': ema_model.state_dict()}),
            'optimizer': optimizer.state_dict(),
            'batch_generator': batch_generator.get_state(),
            'random_state': delu.random.get_state(),
            'early_stopping': early_stopping,
            'report': report,
            'timer': timer,
        }

        return deepcopy(checkpoint)

    print()
    timer.run()
    while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
        if isinstance(optimizer, lib.optim.AdamWScheduleFree):
            optimizer.train()
        model.train()

        batch_losses = []
        batch_sizes = []
        epoch_start_time = timer.elapsed()
        # The `batches` are taken from this example:
        # https://github.com/yandex-research/tabm/blob/main/example.ipynb
        batches = (
            torch.randperm(train_size, generator=batch_generator, device=device).split(
                batch_size
            )
            if share_training_batches
            else (
                torch.rand(
                    (train_size, typing.cast(int, tabm_k)),
                    generator=batch_generator,
                    device=device,
                )
                .argsort(dim=0)
                .split(batch_size, dim=0)
            )
        )
        for batch_idx in tqdm(
            batches,
            desc=str(lib.util.try_get_relative_path(exp)),
            leave=False,
            disable=not lib.env.is_local(),
        ):

            def compute_loss():
                return loss_fn(
                    apply_model(model, dataset, part='train', idx=batch_idx),
                    Y_train[batch_idx],
                )

            loss = compute_loss()
            optimizer.zero_grad()
            loss.backward()

            if gradient_clipping_norm is not None:
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )

            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)

            step += 1
            loss_detached = loss.detach()
            batch_losses.append(loss_detached)
            batch_sizes.append(len(batch_idx))
        epoch_end_time = timer.elapsed()
        batch_losses = torch.stack(batch_losses).tolist()
        epoch_loss = statistics.fmean(batch_losses, weights=batch_sizes)

        (metrics, predictions), eval_batch_size = evaluate(
            parts=['val', 'test'], batch_size=eval_batch_size
        )
        val_score_improved = (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        )
        early_stopping.update(metrics['val']['score'])

        print(
            f'{"*" if val_score_improved else " "}'
            f' [epoch] {step // epoch_size:<3}'
            f' [val] {metrics["val"]["score"]:.3f}'
            f' [test] {metrics["test"]["score"]:.3f}'
            f' [loss] {epoch_loss:.4f}'
            f' [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [it/s] {math.trunc(epoch_size / (epoch_end_time - epoch_start_time)):>3}'
        )
        writer.add_scalars('loss', {'train': epoch_loss}, step, timer.elapsed())
        for part in metrics:
            writer.add_scalars(
                'score', {part: metrics[part]['score']}, step, timer.elapsed()
            )

        if val_score_improved:
            report['best_step'] = step
            report['metrics'] = metrics
            best_checkpoint = make_checkpoint()

        if early_stopping.should_stop() or not lib.util.are_valid_predictions(
            predictions
        ):
            break

    report['time'] = timer.elapsed()

    # >>>
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['model'])
        if ema_model is not None:
            ema_model.load_state_dict(best_checkpoint['ema_model'])
    (metrics, predictions), eval_batch_size = evaluate(
        parts=['train', 'val', 'test'], batch_size=eval_batch_size
    )
    report['eval_batch_size'] = eval_batch_size
    report['metrics'] = metrics
    if config.get('save_checkpoint', False):
        lib.experiment.dump_checkpoint(exp, make_checkpoint())

    lib.experiment.dump_predictions(exp, predictions)
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
