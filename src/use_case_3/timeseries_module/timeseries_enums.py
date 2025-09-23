"""Module for the manipulation of timeseries."""

from __future__ import annotations

from enum import Enum


class AggregationStrategy(Enum):
    """The available aggregation types."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    STD = "std"
    IQR = "IQR"
    NONE = "None"

    def __init__(self, aggregation: str) -> None:
        self.aggregation = aggregation


class AggregationPeriod(Enum):
    """The available aggregation periods."""

    YEAR = ("YS", "Y")
    QUARTER = ("QS", "Q")
    MONTH = ("MS", "M")
    DAY = ("d", "d")

    def __init__(self, period: str, time_period_shorthand: str) -> None:
        self.period = period
        self.time_period_shorthand = time_period_shorthand


class UncertaintyType(Enum):
    """The available uncertainty types."""

    STD = "std"
    IQR = "IQR"
    MAX = "max"
    MIN = "min"
    PVALUE = "p_value"

    def __init__(self, uncertainty_type: str) -> None:
        self.uncertainty_types = uncertainty_type


class NormalisationType(Enum):
    """The available normalisation types."""

    FIRST = "first"
    LINEAR_REGRESSION = "linear_regression"

    def __init__(self, normalisation_type: str) -> None:
        self.normalisation_type = normalisation_type


class TimeseriesType(Enum):
    """Define the type of timeseries. This can be cumulative or derivative."""

    CUMULATIVE = "cumulative"
    DERIVATIVE = "derivative"


class TimeseriesRegion(Enum):
    """
    Define the region type of the timeseries.

    This can be iceshelf, ocean or climate. This defines how the mask is
    calculated and applied.
    """

    ICE_SHELF = "ice_shelf"
    OCEAN = "ocean"
    CLIMATE = "climate"


class ModelEvaluationMetric(Enum):
    """The available model evaluation metrics."""

    AIC = "aic"
    BIC = "bic"

    def __init__(self, evaluation_metric: str) -> None:
        self.evaluation_metric = evaluation_metric


class JointPlotAnalysisType(Enum):
    """The available analysis types used to evaluate correlation or causality between variables."""

    CORR = "corr"
    CROSS_CORR = "cross_corr"
    CAUSAL = "causal"

    def __init__(self, analysis_type: str) -> None:
        self.analysis_type = analysis_type
