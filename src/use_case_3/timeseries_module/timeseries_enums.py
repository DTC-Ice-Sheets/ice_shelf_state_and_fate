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

    def __init__(self, uncertainty_type: str) -> None:
        self.uncertainty_types = uncertainty_type


class NormalisationType(Enum):
    """The available normalisation types."""

    FIRST = "first"
    LINEAR_REGRESSION = "linear_regression"

    def __init__(self, normalisation_type: str) -> None:
        self.normalisation_type = normalisation_type


# class PlotSpreadType(Enum):
#     """The available plot spread types."""

#     NONE = None
#     UNCERTAINTIES = "uncertainties"
#     PERCENTILES = "percentiles"

#     def __init__(self, plotspread_type: str) -> None:
#         self.plotspread_type = plotspread_type


class TimeseriesType(Enum):
    """
    Define the type of timeseries. This can be cumulative or derivative.

    Parameters
    ----------
    Enum : str
        The timeseries type.
    """

    CUMULATIVE = "cumulative"
    DERIVATIVE = "derivative"
