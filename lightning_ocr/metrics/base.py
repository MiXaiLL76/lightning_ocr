from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor
import warnings


class BaseMetric(metaclass=ABCMeta):
    """Base class for a metric.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseMetric` should assign a meaningful value to the
    class attribute `default_prefix`. See the argument `prefix` for details.

    Args:
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    default_prefix: Optional[str] = None

    def __init__(self, prefix: Optional[str] = None) -> None:
        self._dataset_meta: Union[None, dict] = None
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        if self.prefix is None:
            warnings.warn(
                "The prefix is not set in metric class " f"{self.__class__.__name__}."
            )

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int, prefix: str = None) -> dict:
        if len(self.results) == 0:
            warnings.warn(
                f"{self.__class__.__name__} got empty `self.results`. Please "
                "ensure that the processed results are properly added into "
                "`self.results` in `process` method."
            )

        # cast all tensors in results list to cpu
        results = _to_cpu(self.results)
        _metrics = self.compute_metrics(results)  # type: ignore

        # Add prefix to metric names
        if prefix is not None:
            _metrics = {"/".join((prefix, k)): v for k, v in _metrics.items()}
        elif self.prefix:
            _metrics = {"/".join((self.prefix, k)): v for k, v in _metrics.items()}

        metrics = [_metrics]

        # reset the results list
        self.results.clear()
        return metrics[0]


def _to_cpu(data: Any) -> Any:
    """Transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor)):
        return data.to("cpu")
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data
