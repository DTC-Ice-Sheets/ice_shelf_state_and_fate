"""Module for the manipulation of timeseries."""

from __future__ import annotations

from enum import StrEnum


class AggregationStrategy(StrEnum):
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


class AggregationPeriod(StrEnum):
    """The available aggregation periods."""

    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    DAY = "day"

    @staticmethod
    def pd_period_freq(period: AggregationPeriod) -> str:
        """Appropriate value for `pd.PeriodIndex()` `freq` parameter."""
        return {
            AggregationPeriod.YEAR: "Y",
            AggregationPeriod.QUARTER: "Q",
            AggregationPeriod.MONTH: "M",
            AggregationPeriod.DAY: "D",
        }[period]

    @staticmethod
    def pd_date_range_freq(period: AggregationPeriod) -> str:
        """Appropriate value for `pd.date_range()` `freq` parameter."""
        return {
            AggregationPeriod.YEAR: "YS",
            AggregationPeriod.QUARTER: "QS",
            AggregationPeriod.MONTH: "MS",
            AggregationPeriod.DAY: "D",
        }[period]


class UncertaintyType(StrEnum):
    """The available uncertainty types."""

    STD = "std"
    IQR = "IQR"
    MAX = "max"
    MIN = "min"

    def __init__(self, uncertainty_type: str) -> None:
        self.uncertainty_types = uncertainty_type


class NormalisationType(StrEnum):
    """The available normalisation types."""

    FIRST = "first"
    LINEAR_REGRESSION = "linear_regression"

    def __init__(self, normalisation_type: str) -> None:
        self.normalisation_type = normalisation_type


class TimeseriesType(StrEnum):
    """Define the type of timeseries. This can be cumulative or derivative."""

    CUMULATIVE = "cumulative"
    DERIVATIVE = "derivative"


class TimeseriesRegion(StrEnum):
    """
    Define the region type of the timeseries.

    This can be iceshelf, ocean or climate. This defines how the mask is
    calculated and applied.
    """

    ICE_SHELF = "ice_shelf"
    OCEAN = "ocean"
    CLIMATE = "climate"


class ModelEvaluationMetric(StrEnum):
    """The available model evaluation metrics."""

    AIC = "aic"
    BIC = "bic"

    def __init__(self, evaluation_metric: str) -> None:
        self.evaluation_metric = evaluation_metric


class JointPlotAnalysisType(StrEnum):
    """The available analysis types used to evaluate correlation or causality between variables."""

    CORR = "corr"
    CROSS_CORR = "cross_corr"
    CAUSAL = "causal"

    def __init__(self, analysis_type: str) -> None:
        self.analysis_type = analysis_type
