"""Module for undertaking the covariate analysis."""

from typing import Any

import pandas as pd


class CovariateAnalyser:
    """The covariate analysis module."""

    def __init__(self: Any) -> None:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        """
        pass

    def trim_timeseries(self: Any, timeseries: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        timeseries : pd.DataFrame
            _description_
        start_idx : int
            _description_
        end_idx : int
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        pass

    def resample_timeseries(self: Any, timeseries: pd.DataFrame) -> pd.DataFrame:
        """
        _summary_.

        Parameters
        ----------
        self : Any
            _description_
        timeseries : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        pass
