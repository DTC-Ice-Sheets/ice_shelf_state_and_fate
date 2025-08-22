"""Module for creating a timeseries object."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


@dataclass
class Timeseries:
    """The timeseries data class. A wrapped to maintain metadata with the timeseries."""

    label: str
    measurements: pd.Series
    timeseries_type: TimeseriesType
    measurement_uncertainties: pd.Series | None = None
    unit: str = ""


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
