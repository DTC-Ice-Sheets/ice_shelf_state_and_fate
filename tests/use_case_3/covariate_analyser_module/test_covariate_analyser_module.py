"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser


@pytest.fixture
def covariate_analyser_fixture():
    return CovariateAnalyser()


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

    return timeseries


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

    return timeseries


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

    return timeseries


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

    return timeseries


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

    return timeseries


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

    return timeseries


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

    return timeseries


def test_find_start_end_dates(
    covariate_analyser_fixture, timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture
):
    # assert covariate_analyser_fixture.find_start_end_dates([timeseries_1_fixture, timeseries_2_fixture])
    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == covariate_analyser_fixture.find_start_end_dates([timeseries_1_fixture, timeseries_2_fixture])

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == covariate_analyser_fixture.find_start_end_dates([timeseries_1_fixture, timeseries_4_fixture])


def test_check_timeseries_intersection(
    covariate_analyser_fixture, timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture
):
    assert covariate_analyser_fixture.check_timeseries_intersection(
        [timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture]
    )

    assert not covariate_analyser_fixture.check_timeseries_intersection(
        [timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture]
    )


def test_trim_timeseries_date(covariate_analyser_fixture, timeseries_1_fixture):
    timeseries = pd.Series(
        [2, 3, 4],
        index=[
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    assert timeseries.equals(
        covariate_analyser_fixture.trim_timeseries_date(
            timeseries=timeseries_1_fixture,
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )
    )


def test_interpolate_timeseries(covariate_analyser_fixture, timeseries_1_fixture, timeseries_5_fixture):
    ts_interp = covariate_analyser_fixture.interpolate_timeseries(timeseries_5_fixture, period="MS")

    assert np.allclose(timeseries_1_fixture.values, ts_interp.values)
    assert timeseries_1_fixture.index.equals(ts_interp.index)


def test_integrity_check_timeseries(
    covariate_analyser_fixture, timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture
):
    assert covariate_analyser_fixture.integrity_check_timeseries([timeseries_1_fixture, timeseries_1_fixture]) is None

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser_fixture.integrity_check_timeseries([timeseries_1_fixture, timeseries_5_fixture])
    assert str(excinfo.value) == "The time series have different lengths"

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser_fixture.integrity_check_timeseries([timeseries_1_fixture, timeseries_2_fixture])
    assert str(excinfo.value) == "The time series have different indexes"


def test_correlation_coefficient(covariate_analyser_fixture, timeseries_1_fixture, timeseries_2_fixture):
    corr, _ = covariate_analyser_fixture.correlation_coefficient(timeseries_1_fixture, timeseries_2_fixture)
    assert np.isclose(corr, 1)


def test_covariance_normalised(covariate_analyser_fixture, timeseries_1_fixture):
    corr = covariate_analyser_fixture.covariance_normalised([timeseries_1_fixture, timeseries_1_fixture])
    assert np.allclose(corr, np.array([[1, 1], [1, 1]]))


def test_max_crosscorrelation(covariate_analyser_fixture, timeseries_1_fixture, timeseries_1_inv_fixture):
    corr, lag = covariate_analyser_fixture.max_crosscorrelation(timeseries_1_fixture, timeseries_1_fixture)
    assert np.isclose(corr, 1)
    assert lag == 0

    corr, lag = covariate_analyser_fixture.max_crosscorrelation(timeseries_1_fixture, timeseries_1_inv_fixture)
    assert np.isclose(corr, -1)
    assert lag == 0


def test_temporal_offset(covariate_analyser_fixture, timeseries_1_fixture, timeseries_6_fixture):
    assert covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "mean")[0] == 0.4
    assert covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "median")[0] == -1
    assert covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "max")[0] == 4
    assert covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "min")[0] == -1

    assert covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "mean")[1] is None
    assert covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "median")[1] is None
    assert np.array_equal(
        covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "max")[1], 0
    )
    assert np.array_equal(
        covariate_analyser_fixture.temporal_offset(timeseries_1_fixture, timeseries_6_fixture, "min")[1],
        2,
    )


def test_temporal_slope(covariate_analyser_fixture, timeseries_1_fixture, timeseries_1_inv_fixture):
    model_fits = covariate_analyser_fixture.temporal_slope(timeseries_1_fixture, timeseries_1_inv_fixture)
    assert model_fits[0] == -1
    assert model_fits[1] == 1
    assert model_fits[2] == 1
    assert model_fits[3] == -1
    assert model_fits[4] == 5
