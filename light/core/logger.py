from abc import ABC

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only


class ConsoleLogger(LightningLoggerBase, ABC):
    def __init__(self, monitor='val_loss'):
        super().__init__()
        self.monitor = monitor
        # self.mode = mode
        self.value = None

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @property
    @rank_zero_experiment
    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        print(f'\nlog_hyperparams: {params}')

    @rank_zero_only
    def log_metrics(self, metrics, step) -> None:
        metrics = metrics.copy()
        epoch = metrics.pop('epoch')
        metric_text = ', '.join(f'{k}: {v:.3f}' for k, v in metrics.items())
        print(f'  Metrics: epoch {epoch}, step {step}\n{{{metric_text}}}')
        if self.monitor is not None:
            value = metrics[self.monitor]
            if (self.value is None) or (value < self.value):
                self.value = value
                print(f' [Epoch {epoch}], [best {self.monitor}: {self.value:.3f}]')

    @rank_zero_only
    def save(self) -> None:
        super().save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()
