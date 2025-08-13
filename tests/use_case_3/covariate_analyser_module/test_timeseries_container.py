"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

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
def expected_timeseries_dict_fixture1(timeseries_1_fixture):
    return {
        timeseries_1_fixture[1]: [timeseries_1_fixture[0], timeseries_1_fixture[2]],
    }


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
def expected_timeseries_dict_fixture1_fixture_2(timeseries_1_fixture, timeseries_2_fixture):
    return {
        timeseries_1_fixture[1]: [timeseries_1_fixture[0], timeseries_1_fixture[2]],
        timeseries_2_fixture[1]: [timeseries_2_fixture[0], timeseries_2_fixture[2]],
    }


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


def test__init__(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1_fixture_2):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1_fixture_2

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.add_timeseries(
            timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
            timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
            timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
            overwrite=False,
        )
    assert str(excinfo.value) == "Timeseries is already added"


def test_add_timeseries(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1_fixture_2):
    timeseries_container = TimeSeriesContainer(timeseries=[], timeseries_labels=[], timeseries_units=[])

    timeseries_container.add_timeseries(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
        overwrite=True,
    )

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1_fixture_2

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.add_timeseries(
            timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
            timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
            timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
            overwrite=False,
        )
    assert str(excinfo.value) == "Timeseries is already added"


def test__sanity_check_inputs(timeseries_1_fixture):
    assert TimeSeriesContainer._sanity_check_inputs(
        timeseries=[timeseries_1_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2]],
    )

    assert not TimeSeriesContainer._sanity_check_inputs(
        timeseries=[timeseries_1_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries_units=[],
    )


def test_remove_timeseries(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    timeseries_container.remove_timeseries(timeseries_labels=[timeseries_2_fixture[1]])

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.remove_timeseries(timeseries_labels=[timeseries_2_fixture[1]])
    assert str(excinfo.value) == "There is no timeseries with this label"


def test_get_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    timeseries = timeseries_container.get_timeseries(["time series 2"])

    assert timeseries[0].equals(timeseries_2_fixture[0])
    assert timeseries_container.timeseries["time series 2"][0].equals(timeseries[0])

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.get_timeseries(["time series 3"])
    assert str(excinfo.value) == "There is no timeseries with this name"


def test_get_units(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    timeseries_units = timeseries_container.get_units(["time series 2"])

    assert timeseries_units == [timeseries_2_fixture[2]]
    assert [timeseries_container.timeseries["time series 2"][1]] == timeseries_units

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.get_timeseries(["time series 3"])
    assert str(excinfo.value) == "There is no timeseries with this name"


def test_get_timeseries_all(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    timeseries = timeseries_container.get_timeseries_all()

    assert timeseries[0].equals(timeseries_1_fixture[0])
    assert timeseries[1].equals(timeseries_2_fixture[0])


def test_get_labels_all(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    timeseries_labels = timeseries_container.get_labels_all()

    assert timeseries_labels == ["time series 1", "time series 2"]


def test_get_units_all(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    timeseries_labels = timeseries_container.get_units_all()

    assert timeseries_labels == ["unit 1", "unit 2"]


def test_resample_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    # First test the mean
    timeseries_agg = timeseries_container.resample_timeseries(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]], time_period="Y", aggregation="mean"
    )

    assert timeseries_agg[0].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test the median
    timeseries_agg = timeseries_container.resample_timeseries(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]], time_period="Y", aggregation="median"
    )

    assert timeseries_agg[0].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # First test the max
    timeseries_agg = timeseries_container.resample_timeseries(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]], time_period="Y", aggregation="max"
    )

    assert timeseries_agg[0].equals(
        pd.Series(
            [5],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].equals(
        pd.Series(
            [5],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # First test the min
    timeseries_agg = timeseries_container.resample_timeseries(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]], time_period="Y", aggregation="min"
    )

    assert timeseries_agg[0].equals(
        pd.Series(
            [1],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].equals(
        pd.Series(
            [1],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test passing in timeseries directly
    timeseries_agg = timeseries_container.resample_timeseries(
        timeseries_labels=None,
        timeseries=timeseries_container.get_timeseries([timeseries_1_fixture[1], timeseries_2_fixture[1]]),
        time_period="Y",
        aggregation="mean",
    )

    assert timeseries_agg[0].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test passing in both labels and timeseries directly
    timeseries_agg = timeseries_container.resample_timeseries(
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries=timeseries_container.get_timeseries([timeseries_2_fixture[1]]),
        time_period="Y",
        aggregation="mean",
    )

    assert timeseries_agg[0].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )


def test_find_start_end_dates(timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0], timeseries_4_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_4_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2], timeseries_4_fixture[2]],
    )

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]], timeseries=None
    )

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture[1], timeseries_4_fixture[1]], timeseries=None
    )

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates(timeseries_labels=None, timeseries=None)

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture[1]],
        timeseries=timeseries_container.get_timeseries([timeseries_2_fixture[1]]),
    )

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates(
        timeseries_labels=None,
        timeseries=timeseries_container.get_timeseries([timeseries_1_fixture[1], timeseries_2_fixture[1]]),
    )


def test_trim_timeseries_date(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_1 = pd.Series(
        [2, 3, 4],
        index=[
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    timeseries_2 = pd.Series(
        [1, 2, 3],
        index=[
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2]],
    )

    assert timeseries_1.equals(
        timeseries_container.trim_timeseries_date(
            timeseries_labels=[timeseries_1_fixture[1]],
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[0]
    )

    assert timeseries_2.equals(
        timeseries_container.trim_timeseries_date(
            timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1]],
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[1]
    )

    assert timeseries_1.equals(
        timeseries_container.trim_timeseries_date(
            timeseries_labels=None,
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[0]
    )

    assert timeseries_2.equals(
        timeseries_container.trim_timeseries_date(
            timeseries_labels=None,
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[1]
    )

    assert timeseries_1.equals(
        timeseries_container.trim_timeseries_date(
            timeseries=timeseries_container.get_timeseries([timeseries_1_fixture[1], timeseries_2_fixture[1]]),
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[0]
    )

    assert timeseries_2.equals(
        timeseries_container.trim_timeseries_date(
            timeseries=timeseries_container.get_timeseries([timeseries_1_fixture[1], timeseries_2_fixture[1]]),
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[1]
    )


def test_interpolate_timeseries(timeseries_1_fixture, timeseries_5_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_5_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_5_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_5_fixture[2]],
    )

    ts_interp = timeseries_container.interpolate_timeseries(timeseries_labels=[timeseries_5_fixture[1]], period="MS")[0]

    assert np.allclose(timeseries_1_fixture[0].values, ts_interp.values)
    assert timeseries_1_fixture[0].index.equals(ts_interp.index)

    ts_interp = timeseries_container.interpolate_timeseries(
        timeseries=timeseries_container.get_timeseries([timeseries_5_fixture[1]]), period="MS"
    )[0]

    assert np.allclose(timeseries_1_fixture[0].values, ts_interp.values)
    assert timeseries_1_fixture[0].index.equals(ts_interp.index)


def test_prepare_timeseries(timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0], timeseries_5_fixture[0]],
        timeseries_labels=[timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_5_fixture[1]],
        timeseries_units=[timeseries_1_fixture[2], timeseries_2_fixture[2], timeseries_5_fixture[2]],
    )

    timeseries = timeseries_container.prepare_timeseries(time_period="M", aggregation="mean")
    assert timeseries[0].equals(
        pd.Series(
            [2.0, 3, 4, 5],
            index=[
                pd.Timestamp(year=2020, month=2, day=1),
                pd.Timestamp(year=2020, month=3, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
                pd.Timestamp(year=2020, month=5, day=1),
            ],
        )
    )

    assert timeseries[1].equals(
        pd.Series(
            [1, 2.0, 3, 4],
            index=[
                pd.Timestamp(year=2020, month=2, day=1),
                pd.Timestamp(year=2020, month=3, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
                pd.Timestamp(year=2020, month=5, day=1),
            ],
        )
    )

    assert timeseries[2].equals(
        pd.Series(
            [2.0, 3, 4, 5],
            index=[
                pd.Timestamp(year=2020, month=2, day=1),
                pd.Timestamp(year=2020, month=3, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
                pd.Timestamp(year=2020, month=5, day=1),
            ],
        )
    )


def test_check_timeseries_intersection(
    timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture
):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture[0], timeseries_2_fixture[0], timeseries_3_fixture[0], timeseries_4_fixture[0]],
        timeseries_labels=[
            timeseries_1_fixture[1],
            timeseries_2_fixture[1],
            timeseries_3_fixture[1],
            timeseries_4_fixture[1],
        ],
        timeseries_units=[
            timeseries_1_fixture[2],
            timeseries_2_fixture[2],
            timeseries_3_fixture[2],
            timeseries_4_fixture[2],
        ],
    )

    assert timeseries_container.check_timeseries_intersection(
        [timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_3_fixture[1]]
    )

    assert not timeseries_container.check_timeseries_intersection(
        [timeseries_1_fixture[1], timeseries_2_fixture[1], timeseries_4_fixture[1]]
    )
