"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser
from use_case_3.covariate_analyser_module.timeseries_container import TimeSeriesContainer


@pytest.fixture
def timeseries_1_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return timeseries, "time series 1", "unit 1"


@pytest.fixture
def timeseries_2_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
            pd.Timestamp(year=2020, month=6, day=1),
        ],
    )

    return timeseries, "time series 2", "unit 2"


@pytest.fixture
def timeseries_3_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
            pd.Timestamp(year=2020, month=6, day=1),
            pd.Timestamp(year=2020, month=7, day=1),
        ],
    )

    return timeseries, "time series 3", "unit 3"


@pytest.fixture
def timeseries_4_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=7, day=1),
            pd.Timestamp(year=2020, month=8, day=1),
            pd.Timestamp(year=2020, month=9, day=1),
            pd.Timestamp(year=2020, month=10, day=1),
            pd.Timestamp(year=2020, month=11, day=1),
        ],
    )

    return timeseries, "time series 4", "unit 4"


@pytest.fixture
def timeseries_5_fixture():
    timeseries = pd.Series(
        [1, 2, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return timeseries, "time series 5", "unit 5"


@pytest.fixture
def timeseries_1_inv_fixture():
    timeseries = pd.Series(
        [5, 4, 3, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return timeseries, "time series 1 inv", "unit 1 inv"


@pytest.fixture
def timeseries_6_fixture():
    timeseries = pd.Series(
        [-3, 1, 4, 5, 6],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return timeseries, "time series 6", "unit 6"


@pytest.fixture
def timeseries_7_fixture():
    timeseries = pd.Series(
        [0, 1, 2, 1, 0],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return timeseries, "time series 7", "unit 7"


@pytest.fixture
def timeseries_7_lag_fixture():
    timeseries = pd.Series(
        [1, 0, 1, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )
    return timeseries, "time series 7 inv", "unit 7 inv"


def test_correlation_coefficient(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    covariate_analyser = CovariateAnalyser(timeseries=timeseries_container)
    corr, _ = covariate_analyser.correlation_coefficient(timeseries_1_fixture[1], timeseries_2_fixture[1])
    assert np.isclose(corr, 1)


def test_covariance(timeseries_1_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2]],
    )

    covariate_analyser = CovariateAnalyser(timeseries=timeseries_container)
    corr = covariate_analyser.covariance([timeseries_1_fixture[1], timeseries_1_fixture[1]], normalisation="corrcoef")
    assert np.allclose(corr, np.array([[1, 1], [1, 1]]))


def test_max_crosscorrelation(
    timeseries_1_fixture, timeseries_1_inv_fixture, timeseries_7_fixture, timeseries_7_lag_fixture
):
    timeseries_container = TimeSeriesContainer(
        timeseries=[
            timeseries_1_fixture[0],
            timeseries_1_inv_fixture[0],
            timeseries_7_fixture[0],
            timeseries_7_lag_fixture[0],
        ],
        timeseries_labels=[
            timeseries_1_fixture[1],
            timeseries_1_inv_fixture[1],
            timeseries_7_fixture[1],
            timeseries_7_lag_fixture[1],
        ],
        timeseries_units=[
            timeseries_1_fixture[2],
            timeseries_1_inv_fixture[2],
            timeseries_7_fixture[2],
            timeseries_7_lag_fixture[2],
        ],
    )

    covariate_analyser = CovariateAnalyser(timeseries=timeseries_container)

    corr, lag = covariate_analyser.max_crosscorrelation(timeseries_1_fixture[1], timeseries_1_fixture[1], max_lag=None)
    assert np.isclose(corr, 1)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_1_fixture[1], timeseries_1_inv_fixture[1], max_lag=None
    )
    assert np.isclose(corr, -1)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(timeseries_7_fixture[1], timeseries_7_lag_fixture[1], max_lag=0)

    assert np.isclose(corr, 0)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(timeseries_7_fixture[1], timeseries_7_lag_fixture[1], max_lag=1)

    assert np.isclose(corr, 0.84515425)
    assert lag == -1


def test_align_timeseries(timeseries_7_fixture, timeseries_7_lag_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[
            timeseries_7_fixture[0],
            timeseries_7_lag_fixture[0],
        ],
        timeseries_labels=[
            timeseries_7_fixture[1],
            timeseries_7_lag_fixture[1],
        ],
        timeseries_units=[
            timeseries_7_fixture[2],
            timeseries_7_lag_fixture[2],
        ],
    )

    covariate_analyser = CovariateAnalyser(timeseries=timeseries_container)

    timeseries_1, timeseries_2, freq, corr_lag = covariate_analyser.align_timeseries(
        timeseries_label_1=timeseries_7_fixture[1], timeseries_label_2=timeseries_7_lag_fixture[1], max_lag=1
    )

    timeseries_1_gt = pd.Series(
        [0, 1, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    timeseries_2_gt = pd.Series(
        [0, 1, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    assert np.allclose(timeseries_1.values, timeseries_1_gt.values)
    assert np.allclose(timeseries_2.values, timeseries_2_gt.values)
    assert freq == "Month"
    assert corr_lag == -1


def test_temporal_offset(timeseries_1_fixture, timeseries_6_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[
            timeseries_1_fixture[0],
            timeseries_6_fixture[0],
        ],
        timeseries_labels=[
            timeseries_1_fixture[1],
            timeseries_6_fixture[1],
        ],
        timeseries_units=[
            timeseries_1_fixture[2],
            timeseries_6_fixture[2],
        ],
    )

    covariate_analyser = CovariateAnalyser(timeseries=timeseries_container)
    assert covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "mean")[0] == 0.4
    assert covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "median")[0] == -1
    assert covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "max")[0] == 4
    assert covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "min")[0] == -1

    assert covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "mean")[1] is None
    assert covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "median")[1] is None
    assert np.array_equal(
        covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "max")[1], 0
    )
    assert np.array_equal(
        covariate_analyser.temporal_offset(timeseries_1_fixture[1], timeseries_6_fixture[1], "min")[1],
        2,
    )


def test_temporal_slope(timeseries_1_fixture, timeseries_1_inv_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[
            timeseries_1_fixture[0],
            timeseries_1_inv_fixture[0],
        ],
        timeseries_labels=[
            timeseries_1_fixture[1],
            timeseries_1_inv_fixture[1],
        ],
        timeseries_units=[
            timeseries_1_fixture[2],
            timeseries_1_inv_fixture[2],
        ],
    )

    covariate_analyser = CovariateAnalyser(timeseries=timeseries_container)
    model_fits = covariate_analyser.temporal_slope(timeseries_1_fixture[1], timeseries_1_inv_fixture[1])
    assert model_fits[0] == -1
    assert model_fits[1] == 1
    assert model_fits[2] == 1
    assert model_fits[3] == -1
    assert model_fits[4] == 5
