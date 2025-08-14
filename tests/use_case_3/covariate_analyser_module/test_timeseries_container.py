"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.timeseries_container import (
    AggregationPeriod,
    AggregationStrategy,
    TimeSeriesContainer,
)
from use_case_3.timeseries_module.timeseries import Timeseries, TimeseriesType


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

    return Timeseries(
        measurements=timeseries, label="time series 1", unit="unit 1", timeseries_type=TimeseriesType("cumulative")
    )


@pytest.fixture
def expected_timeseries_dict_fixture1(timeseries_1_fixture):
    return {
        timeseries_1_fixture.label: timeseries_1_fixture,
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

    return Timeseries(
        measurements=timeseries, label="time series 2", unit="unit 2", timeseries_type=TimeseriesType("cumulative")
    )


@pytest.fixture
def expected_timeseries_dict_fixture1_fixture_2(timeseries_1_fixture, timeseries_2_fixture):
    return {
        timeseries_1_fixture.label: timeseries_1_fixture,
        timeseries_2_fixture.label: timeseries_2_fixture,
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

    return Timeseries(
        measurements=timeseries, label="time series 3", unit="unit 3", timeseries_type=TimeseriesType("cumulative")
    )


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

    return Timeseries(
        measurements=timeseries, label="time series 4", unit="unit 4", timeseries_type=TimeseriesType("cumulative")
    )


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

    return Timeseries(
        measurements=timeseries, label="time series 5", unit="unit 5", timeseries_type=TimeseriesType("cumulative")
    )


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

    return Timeseries(
        measurements=timeseries,
        label="time series 1 inv",
        unit="unit 1 inv",
        timeseries_type=TimeseriesType("cumulative"),
    )


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

    return Timeseries(
        measurements=timeseries, label="time series 6", unit="unit 6", timeseries_type=TimeseriesType("cumulative")
    )


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

    return Timeseries(
        measurements=timeseries, label="time series 7", unit="unit 7", timeseries_type=TimeseriesType("cumulative")
    )


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
    return Timeseries(
        measurements=timeseries,
        label="time series 7 inv",
        unit="unit 7 inv",
        timeseries_type=TimeseriesType("cumulative"),
    )


def test__init__(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1_fixture_2):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1_fixture_2

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.add_timeseries(
            timeseries=[timeseries_1_fixture, timeseries_2_fixture],
            overwrite=False,
        )
    assert str(excinfo.value) == "Timeseries is already added"


def test_add_timeseries(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1_fixture_2):
    timeseries_container = TimeSeriesContainer(timeseries=[])

    timeseries_container.add_timeseries(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        overwrite=True,
    )

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1_fixture_2

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.add_timeseries(
            timeseries=[timeseries_1_fixture, timeseries_2_fixture],
            overwrite=False,
        )
    assert str(excinfo.value) == "Timeseries is already added"


def test_remove_timeseries(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    timeseries_container.remove_timeseries(timeseries_labels=[timeseries_2_fixture.label])

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.remove_timeseries(timeseries_labels=[timeseries_2_fixture.label])
    assert str(excinfo.value) == "There is no timeseries with this label"


def test_get_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    timeseries = timeseries_container.get_timeseries(["time series 2"])

    assert timeseries[0] == timeseries_2_fixture
    assert timeseries_container.timeseries["time series 2"] == timeseries[0]

    with pytest.raises(ValueError) as excinfo:
        timeseries_container.get_timeseries(["time series 3"])
    assert str(excinfo.value) == "There is no timeseries with this name"


def test_get_timeseries_all(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    timeseries = timeseries_container.get_timeseries_all()

    assert timeseries[0] == timeseries_1_fixture
    assert timeseries[1] == timeseries_2_fixture


def test_get_labels_all(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    timeseries_labels = timeseries_container.get_labels_all()

    assert timeseries_labels == ["time series 1", "time series 2"]


def test__resample_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    # First test the mean
    timeseries_agg = timeseries_container._resample_timeseries(
        timeseries_labels=[timeseries_1_fixture.label, timeseries_2_fixture.label],
        time_period="Y",
        aggregation=AggregationStrategy("mean"),
    )

    assert timeseries_agg[0].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test the median
    timeseries_agg = timeseries_container._resample_timeseries(
        timeseries_labels=[timeseries_1_fixture.label, timeseries_2_fixture.label],
        time_period="Y",
        aggregation=AggregationStrategy("median"),
    )

    assert timeseries_agg[0].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # First test the max
    timeseries_agg = timeseries_container._resample_timeseries(
        timeseries_labels=[timeseries_1_fixture.label, timeseries_2_fixture.label],
        time_period="Y",
        aggregation=AggregationStrategy("max"),
    )

    assert timeseries_agg[0].measurements.equals(
        pd.Series(
            [5],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].measurements.equals(
        pd.Series(
            [5],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # First test the min
    timeseries_agg = timeseries_container._resample_timeseries(
        timeseries_labels=[timeseries_1_fixture.label, timeseries_2_fixture.label],
        time_period="Y",
        aggregation=AggregationStrategy("min"),
    )

    assert timeseries_agg[0].measurements.equals(
        pd.Series(
            [1],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].measurements.equals(
        pd.Series(
            [1],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test passing in timeseries directly
    timeseries_agg = timeseries_container._resample_timeseries(
        timeseries_labels=None,
        timeseries=timeseries_container.get_timeseries([timeseries_1_fixture.label, timeseries_2_fixture.label]),
        time_period="Y",
        aggregation=AggregationStrategy("mean"),
    )

    assert timeseries_agg[0].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test passing in both labels and timeseries directly
    timeseries_agg = timeseries_container._resample_timeseries(
        timeseries_labels=[timeseries_1_fixture.label],
        timeseries=timeseries_container.get_timeseries([timeseries_2_fixture.label]),
        time_period="Y",
        aggregation=AggregationStrategy("mean"),
    )

    assert timeseries_agg[0].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_agg[1].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )


def test__find_start_end_dates(timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture],
    )

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container._find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture.label, timeseries_2_fixture.label], timeseries=None
    )

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container._find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture.label, timeseries_4_fixture.label], timeseries=None
    )

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container._find_start_end_dates(timeseries_labels=None, timeseries=None)

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container._find_start_end_dates(
        timeseries_labels=[timeseries_1_fixture.label],
        timeseries=timeseries_container.get_timeseries([timeseries_2_fixture.label]),
    )

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container._find_start_end_dates(
        timeseries_labels=None,
        timeseries=timeseries_container.get_timeseries([timeseries_1_fixture.label, timeseries_2_fixture.label]),
    )


def test__trim_timeseries_date(timeseries_1_fixture, timeseries_2_fixture):
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
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    assert timeseries_1.equals(
        timeseries_container._trim_timeseries_date(
            timeseries_labels=[timeseries_1_fixture.label],
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[0].measurements
    )

    assert timeseries_2.equals(
        timeseries_container._trim_timeseries_date(
            timeseries_labels=[timeseries_1_fixture.label, timeseries_2_fixture.label],
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[1].measurements
    )

    assert timeseries_1.equals(
        timeseries_container._trim_timeseries_date(
            timeseries_labels=None,
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[0].measurements
    )

    assert timeseries_2.equals(
        timeseries_container._trim_timeseries_date(
            timeseries_labels=None,
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[1].measurements
    )

    assert timeseries_1.equals(
        timeseries_container._trim_timeseries_date(
            timeseries=timeseries_container.get_timeseries([timeseries_1_fixture.label, timeseries_2_fixture.label]),
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[0].measurements
    )

    assert timeseries_2.equals(
        timeseries_container._trim_timeseries_date(
            timeseries=timeseries_container.get_timeseries([timeseries_1_fixture.label, timeseries_2_fixture.label]),
            start_date=pd.Timestamp(year=2020, month=2, day=1),
            end_date=pd.Timestamp(year=2020, month=4, day=1),
        )[1].measurements
    )


def test__interpolate_timeseries(timeseries_1_fixture, timeseries_5_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_5_fixture],
    )

    ts_interp = timeseries_container._interpolate_timeseries(
        timeseries_labels=[timeseries_5_fixture.label], period="MS"
    )[0]

    assert np.allclose(timeseries_1_fixture.measurements.values, ts_interp.measurements.values)
    assert timeseries_1_fixture.measurements.index.equals(ts_interp.measurements.index)

    ts_interp = timeseries_container._interpolate_timeseries(
        timeseries=timeseries_container.get_timeseries([timeseries_5_fixture.label]), period="MS"
    )[0]

    assert np.allclose(timeseries_1_fixture.measurements.values, ts_interp.measurements.values)
    assert timeseries_1_fixture.measurements.index.equals(ts_interp.measurements.index)


def test_prepare_timeseries(timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture],
    )

    timeseries = timeseries_container.prepare_timeseries(
        time_period=AggregationPeriod.MONTH, aggregation=AggregationStrategy("mean"), inplace=False
    )
    assert timeseries[0].measurements.equals(
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

    assert timeseries[1].measurements.equals(
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

    assert timeseries[2].measurements.equals(
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


def test__check_timeseries_intersection(
    timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture
):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture],
    )

    assert timeseries_container._check_timeseries_intersection(
        [timeseries_1_fixture.label, timeseries_2_fixture.label, timeseries_3_fixture.label]
    )

    assert not timeseries_container._check_timeseries_intersection(
        [timeseries_1_fixture.label, timeseries_2_fixture.label, timeseries_4_fixture.label]
    )


def test__combine_timeseries_from_labels_and_list(
    timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture
):
    timeseries_container = TimeSeriesContainer(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture, timeseries_4_fixture],
    )

    timeseries = timeseries_container._combine_timeseries_from_labels_and_list(
        timeseries_labels=[
            timeseries_1_fixture.label,
            timeseries_2_fixture.label,
            timeseries_3_fixture.label,
            timeseries_4_fixture.label,
        ]
    )
    assert timeseries[0] == timeseries_1_fixture
    assert timeseries[1] == timeseries_2_fixture
    assert timeseries[2] == timeseries_3_fixture
    assert timeseries[3] == timeseries_4_fixture

    timeseries = timeseries_container._combine_timeseries_from_labels_and_list(
        timeseries_labels=[
            timeseries_1_fixture.label,
            timeseries_2_fixture.label,
        ],
        timeseries=[
            timeseries_3_fixture,
            timeseries_4_fixture,
        ],
    )
    assert timeseries[0] == timeseries_3_fixture
    assert timeseries[1] == timeseries_4_fixture
    assert timeseries[2] == timeseries_1_fixture
    assert timeseries[3] == timeseries_2_fixture

    timeseries = timeseries_container._combine_timeseries_from_labels_and_list(
        timeseries=[
            timeseries_1_fixture,
            timeseries_2_fixture,
            timeseries_3_fixture,
            timeseries_4_fixture,
        ]
    )
    assert timeseries[0] == timeseries_1_fixture
    assert timeseries[1] == timeseries_2_fixture
    assert timeseries[2] == timeseries_3_fixture
    assert timeseries[3] == timeseries_4_fixture
