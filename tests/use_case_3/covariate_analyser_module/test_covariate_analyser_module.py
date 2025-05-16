"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.covariate_analyser_module import CovariateAnalyser


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


def test_find_start_end_dates(timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
            timeseries_4_fixture[1]: timeseries_4_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_4_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
            timeseries_4_fixture[1]: timeseries_4_fixture[2],
        },
    )

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == covariate_analyser.find_start_end_dates(timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]])

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == covariate_analyser.find_start_end_dates(timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]])

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == covariate_analyser.find_start_end_dates(timeseries_labels=[timeseries_1_fixture[1], timeseries_4_fixture[1]])

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == covariate_analyser.find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]], timeseries=[timeseries_4_fixture[0]]
    )

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser.find_start_end_dates(timeseries_labels=None, timeseries=None)
    assert str(excinfo.value) == "Both the list of labels and timeseries can't be None"


def test_check_timeseries_intersection(
    timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture
):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
            timeseries_3_fixture[1]: timeseries_3_fixture[0],
            timeseries_4_fixture[1]: timeseries_4_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_4_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
            timeseries_3_fixture[1]: timeseries_3_fixture[2],
            timeseries_4_fixture[1]: timeseries_4_fixture[2],
        },
    )

    assert covariate_analyser.check_timeseries_intersection(
        [timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_3_fixture[1]]
    )

    assert not covariate_analyser.check_timeseries_intersection(
        [timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_4_fixture[1]]
    )


def test_trim_timeseries_date(timeseries_1_fixture):
    timeseries = pd.Series(
        [2, 3, 4],
        index=[
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
        },
    )

    assert timeseries.equals(
        covariate_analyser.trim_timeseries_date(
            timeseries_label=timeseries_1_fixture[1],
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )
    )

    assert timeseries.equals(
        covariate_analyser.trim_timeseries_date(
            timeseries=timeseries_1_fixture[0],
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )
    )


def test_interpolate_timeseries(timeseries_1_fixture, timeseries_5_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_5_fixture[1]: timeseries_5_fixture[0],
        },
        timeseries_labels=[timeseries_5_fixture[1]],
        timeseries_units={
            timeseries_5_fixture[1]: timeseries_5_fixture[2],
        },
    )

    ts_interp = covariate_analyser.interpolate_timeseries(timeseries_label=timeseries_5_fixture[1], period="MS")

    assert np.allclose(timeseries_1_fixture[0].values, ts_interp.values)
    assert timeseries_1_fixture[0].index.equals(ts_interp.index)


def test_integrity_check_timeseries(timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
            timeseries_5_fixture[1]: timeseries_5_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_5_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
            timeseries_5_fixture[1]: timeseries_5_fixture[2],
        },
    )
    assert covariate_analyser.integrity_check_timeseries([timeseries_1_fixture[1], timeseries_1_fixture[1]]) is None

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser.integrity_check_timeseries([timeseries_1_fixture[1], timeseries_5_fixture[1]])
    assert str(excinfo.value) == "The time series have different lengths"

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser.integrity_check_timeseries([timeseries_1_fixture[1], timeseries_2_fixture[1]])
    assert str(excinfo.value) == "The time series have different indexes"


def test_correlation_coefficient_label(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )
    corr, _ = covariate_analyser.correlation_coefficient_label(timeseries_1_fixture[1], timeseries_2_fixture[1])
    assert np.isclose(corr, 1)


def test_correlation_coefficient(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )
    corr, _ = covariate_analyser.correlation_coefficient(timeseries_1_fixture[0], timeseries_2_fixture[0])
    assert np.isclose(corr, 1)


def test_covariance_normalised(timeseries_1_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
        },
    )
    corr = covariate_analyser.covariance_normalised([timeseries_1_fixture[1], timeseries_1_fixture[1]])
    assert np.allclose(corr, np.array([[1, 1], [1, 1]]))


def test_max_crosscorrelation(
    timeseries_1_fixture, timeseries_1_inv_fixture, timeseries_7_fixture, timeseries_7_lag_fixture
):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_1_inv_fixture[1]: timeseries_1_inv_fixture[0],
            timeseries_7_fixture[1]: timeseries_7_fixture[0],
            timeseries_7_lag_fixture[1]: timeseries_7_lag_fixture[0],
        },
        timeseries_labels=[
            timeseries_1_fixture[1],
            timeseries_1_inv_fixture[1],
            timeseries_7_fixture[1],
            timeseries_7_lag_fixture[1],
        ],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_1_inv_fixture[1]: timeseries_1_inv_fixture[2],
            timeseries_7_fixture[1]: timeseries_7_fixture[2],
            timeseries_7_lag_fixture[1]: timeseries_7_lag_fixture[2],
        },
    )
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
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_7_fixture[1]: timeseries_7_fixture[0],
            timeseries_7_lag_fixture[1]: timeseries_7_lag_fixture[0],
        },
        timeseries_labels=[
            timeseries_7_fixture[1],
            timeseries_7_lag_fixture[1],
        ],
        timeseries_units={
            timeseries_7_fixture[1]: timeseries_7_fixture[2],
            timeseries_7_lag_fixture[1]: timeseries_7_lag_fixture[2],
        },
    )

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

    print(timeseries_1)
    print(timeseries_1_gt)

    print(timeseries_2)
    print(timeseries_2_gt)

    assert np.allclose(timeseries_1.values, timeseries_1_gt.values)
    assert np.allclose(timeseries_2.values, timeseries_2_gt.values)
    assert freq == "Month"
    assert corr_lag == -1


def test_temporal_offset(timeseries_1_fixture, timeseries_6_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_6_fixture[1]: timeseries_6_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_6_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_6_fixture[1]: timeseries_6_fixture[2],
        },
    )
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
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_1_inv_fixture[1]: timeseries_1_inv_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_1_inv_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_1_inv_fixture[1]: timeseries_1_inv_fixture[2],
        },
    )
    model_fits = covariate_analyser.temporal_slope(timeseries_1_fixture[1], timeseries_1_inv_fixture[1])
    assert model_fits[0] == -1
    assert model_fits[1] == 1
    assert model_fits[2] == 1
    assert model_fits[3] == -1
    assert model_fits[4] == 5


def test_set_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={},
        timeseries_labels=[],
        timeseries_units={},
    )

    covariate_analyser.set_timeseries(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )
    assert covariate_analyser.timeseries_labels == ["time series 1", "time series 2"]
    assert covariate_analyser.timeseries["time series 1"].equals(timeseries_1_fixture[0])
    assert covariate_analyser.timeseries["time series 2"].equals(timeseries_2_fixture[0])
    assert covariate_analyser.timeseries_units == {"time series 1": "unit 1", "time series 2": "unit 2"}


def test_add_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
        },
    )

    covariate_analyser.add_timeseries(
        timeseries=timeseries_2_fixture[0],
        timeseries_label=timeseries_2_fixture[1],
        timeseries_unit=timeseries_2_fixture[2],
    )

    assert covariate_analyser.timeseries_labels == ["time series 1", "time series 2"]
    assert covariate_analyser.timeseries["time series 1"].equals(timeseries_1_fixture[0])
    assert covariate_analyser.timeseries["time series 2"].equals(timeseries_2_fixture[0])
    assert covariate_analyser.timeseries_units == {"time series 1": "unit 1", "time series 2": "unit 2"}


def test_remove_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )

    covariate_analyser.remove_timeseries(timeseries_label=timeseries_2_fixture[1])

    assert covariate_analyser.timeseries_labels == ["time series 1"]
    assert covariate_analyser.timeseries["time series 1"].equals(timeseries_1_fixture[0])
    assert covariate_analyser.timeseries_units == {"time series 1": "unit 1"}

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser.remove_timeseries(timeseries_label=timeseries_2_fixture[1])
    assert str(excinfo.value) == "There is no timeseries with this name"


def test_get_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )

    timeseries = covariate_analyser.get_timeseries("time series 2")

    assert timeseries.equals(timeseries_2_fixture[0])
    assert covariate_analyser.timeseries["time series 2"].equals(timeseries)

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser.get_timeseries("time series 3")
    assert str(excinfo.value) == "There is no timeseries with this name"


def test_get_units(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )

    timeseries_units = covariate_analyser.get_units("time series 2")

    assert timeseries_units == timeseries_2_fixture[2]
    assert covariate_analyser.timeseries_units["time series 2"] == timeseries_units

    with pytest.raises(ValueError) as excinfo:
        covariate_analyser.get_units("time series 3")
    assert str(excinfo.value) == "There is no timeseries with this name"


def test_get_timeseries_all(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )

    timeseries = covariate_analyser.get_timeseries_all()

    assert timeseries["time series 1"].equals(timeseries_1_fixture[0])
    assert timeseries["time series 2"].equals(timeseries_2_fixture[0])


def test_get_labels_all(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )

    timeseries_labels = covariate_analyser.get_labels_all()

    assert timeseries_labels == ["time series 1", "time series 2"]


def test_get_units_all(timeseries_1_fixture, timeseries_2_fixture):
    covariate_analyser = CovariateAnalyser(
        timeseries={
            timeseries_1_fixture[1]: timeseries_1_fixture[0],
            timeseries_2_fixture[1]: timeseries_2_fixture[0],
        },
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units={
            timeseries_1_fixture[1]: timeseries_1_fixture[2],
            timeseries_2_fixture[1]: timeseries_2_fixture[2],
        },
    )

    timeseries_labels = covariate_analyser.get_units_all()

    assert timeseries_labels == {"time series 1": "unit 1", "time series 2": "unit 2"}
