"""Module for processing data into timeseries."""

import numpy as np


class TimeSeriesModule:
    """The time series module for calculating and determining statistics."""

    def __init__(self) -> None:
        """
        Needs to be able to return summary statistics with different aggregations in time. Needs to be able
        to process and combine uncertainties.
        """
        pass

    @staticmethod
    def calculate_mean(data: np.ndarray, dims: dict) -> np.ndarray:
        """
        Calculate the mean along the none time axes resulting in a 1d timeseries of mean values with associated
        uncertainties.

        Parameters
        ----------
        data : np.ndarray
            The input n-dim data array. When the array is 1d along the time axis this will be returned with no
            aggregation.
        dims : dict
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        pass

    @staticmethod
    def calculate_median(data: np.ndarray, dims: dict) -> np.ndarray:
        """
        Calculate the median along the none time axes resulting in a 1d timeseries of mean values with associated
        uncertainties.

        Parameters
        ----------
        data : np.ndarray
            The input n-dim data array. When the array is 1d along the time axis this will be returned with no
            aggregation.
        dims : dict
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        pass
