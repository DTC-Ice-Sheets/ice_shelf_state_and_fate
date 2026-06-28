"""Test the covariate analyser module."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.timeseries_preprocessor import (
    TimeSeriesPreprocessor,
)
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
)


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
        measurements=timeseries,
        label="time series 1",
        unit="unit 1",
        measurement_name="measurement 1",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurements=timeseries,
        label="time series 2",
        unit="unit 2",
        measurement_name="measurement 2",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurements=timeseries,
        label="time series 3",
        unit="unit 3",
        measurement_name="measurement 3",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurements=timeseries,
        label="time series 4",
        unit="unit 4",
        measurement_name="measurement 4",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurements=timeseries,
        label="time series 5",
        unit="unit 5",
        measurement_name="measurement 5",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurement_name="measurement 1 inv",
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
        measurements=timeseries,
        label="time series 6",
        unit="unit 6",
        measurement_name="measurement 6",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurements=timeseries,
        label="time series 7",
        unit="unit 7",
        measurement_name="measurement 7",
        timeseries_type=TimeseriesType("cumulative"),
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
        measurement_name="measurement 7",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_merge_fixture():
    timeseries_1 = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    timeseries_2 = pd.Series(
        [3, 4, 5, 6, 7],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return [
        Timeseries(
            measurements=timeseries_1,
            label="time series merge",
            unit="unit 1",
            measurement_name="measurement 1",
            timeseries_type=TimeseriesType("cumulative"),
        ),
        Timeseries(
            measurements=timeseries_2,
            label="time series merge",
            unit="unit 1",
            measurement_name="measurement 1",
            timeseries_type=TimeseriesType("cumulative"),
        ),
    ]


def test__init__(
    timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture, expected_timeseries_dict_fixture1_fixture_2
):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1_fixture_2

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
    )

    assert timeseries_container.timeseries[timeseries_1_fixture.label].measurements.equals(
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

    assert timeseries_container.timeseries[timeseries_2_fixture.label].measurements.equals(
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

    assert timeseries_container.timeseries[timeseries_5_fixture.label].measurements.equals(
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


@pytest.mark.parametrize(
    ("times", "expected_period"),
    [
        (
            [pd.Timestamp(year=2020, month=1, day=1), pd.Timestamp(year=2020, month=1, day=15)],
            AggregationPeriod.DAY,
        ),
        (
            [pd.Timestamp(year=2020, month=1, day=1), pd.Timestamp(year=2020, month=2, day=1)],
            AggregationPeriod.MONTH,
        ),
        (
            [pd.Timestamp(year=2020, month=1, day=1), pd.Timestamp(year=2020, month=3, day=31)],
            AggregationPeriod.QUARTER,
        ),
        (
            [pd.Timestamp(year=2020, month=1, day=1), pd.Timestamp(year=2021, month=1, day=1)],
            AggregationPeriod.YEAR,
        ),
        (
            [
                pd.Timestamp(year=2020, month=1, day=1),
                pd.Timestamp(year=2020, month=3, day=31),
                pd.Timestamp(year=2020, month=4, day=1),
            ],
            AggregationPeriod.QUARTER,
        ),
    ],
)
def test_default_time_period(times, expected_period):
    mock_timeseries = MagicMock()
    mock_timeseries.measurements.index = pd.DatetimeIndex(times)
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[mock_timeseries],
        time_period=None,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )
    assert timeseries_container.aggregation_period == expected_period


def test_add_timeseries(timeseries_1_fixture, timeseries_2_fixture, expected_timeseries_dict_fixture1_fixture_2):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container.add_timeseries(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
    )

    assert timeseries_container.timeseries == expected_timeseries_dict_fixture1_fixture_2


def test_resample_timeseries(timeseries_1_fixture, timeseries_2_fixture):
    # First test the mean
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.YEAR,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container._resample_timeseries()

    assert timeseries_container.timeseries[timeseries_1_fixture.label].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_container.timeseries[timeseries_2_fixture.label].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # Test the median
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.YEAR,
        aggregation=AggregationStrategy.MEDIAN,
        process_timeseries=False,
    )

    timeseries_container._resample_timeseries()

    assert timeseries_container.timeseries[timeseries_1_fixture.label].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_container.timeseries[timeseries_2_fixture.label].measurements.equals(
        pd.Series(
            [3.0],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # First test the max
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.YEAR,
        aggregation=AggregationStrategy.MAX,
        process_timeseries=False,
    )

    timeseries_container._resample_timeseries()

    assert timeseries_container.timeseries[timeseries_1_fixture.label].measurements.equals(
        pd.Series(
            [5],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_container.timeseries[timeseries_2_fixture.label].measurements.equals(
        pd.Series(
            [5],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    # First test the min
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.YEAR,
        aggregation=AggregationStrategy.MIN,
        process_timeseries=False,
    )

    timeseries_container._resample_timeseries()

    assert timeseries_container.timeseries[timeseries_1_fixture.label].measurements.equals(
        pd.Series(
            [1],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )

    assert timeseries_container.timeseries[timeseries_2_fixture.label].measurements.equals(
        pd.Series(
            [1],
            index=[
                pd.Timestamp(year=2020, month=1, day=1),
            ],
        )
    )


def test_find_start_end_dates(timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    assert (
        pd.Timestamp(year=2020, month=2, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates()

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_4_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    assert (
        pd.Timestamp(year=2020, month=7, day=1),
        pd.Timestamp(year=2020, month=5, day=1),
    ) == timeseries_container.find_start_end_dates()


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

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container.trim_start_end_dates(
        start_date=pd.Timestamp(year=2020, month=2, day=1),
        end_date=pd.Timestamp(year=2020, month=4, day=1),
    )

    assert timeseries_1.equals(timeseries_container.timeseries[timeseries_1_fixture.label].measurements)

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container.trim_start_end_dates(
        start_date=pd.Timestamp(year=2020, month=2, day=1),
        end_date=pd.Timestamp(year=2020, month=4, day=1),
    )

    assert timeseries_2.equals(timeseries_container.timeseries[timeseries_2_fixture.label].measurements)

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container.trim_start_end_dates(
        start_date=pd.Timestamp(year=2020, month=2, day=1),
        end_date=pd.Timestamp(year=2020, month=4, day=1),
    )

    assert timeseries_1.equals(timeseries_container.timeseries[timeseries_1_fixture.label].measurements)

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container.trim_start_end_dates(
        start_date=pd.Timestamp(year=2020, month=2, day=1),
        end_date=pd.Timestamp(year=2020, month=4, day=1),
    )

    assert timeseries_2.equals(timeseries_container.timeseries[timeseries_2_fixture.label].measurements)


def test_interpolate_timeseries(timeseries_1_fixture, timeseries_5_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_5_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container._interpolate_timeseries()

    assert np.allclose(
        timeseries_1_fixture.measurements.values,
        timeseries_container.timeseries[timeseries_1_fixture.label].measurements.values,
    )
    assert timeseries_1_fixture.measurements.index.equals(
        timeseries_container.timeseries[timeseries_1_fixture.label].measurements.index
    )

    assert np.allclose(
        timeseries_1_fixture.measurements.values,
        timeseries_container.timeseries[timeseries_5_fixture.label].measurements.values,
    )
    assert timeseries_1_fixture.measurements.index.equals(
        timeseries_container.timeseries[timeseries_5_fixture.label].measurements.index
    )


def test_process_timeseries(timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_5_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container.process_timeseries()
    assert timeseries_container.timeseries[timeseries_1_fixture.label].measurements.equals(
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

    assert timeseries_container.timeseries[timeseries_2_fixture.label].measurements.equals(
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

    assert timeseries_container.timeseries[timeseries_5_fixture.label].measurements.equals(
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
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_3_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    assert timeseries_container._check_timeseries_intersection()

    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_4_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    assert not timeseries_container._check_timeseries_intersection()


def test_merge_timeseries(timeseries_merge_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=timeseries_merge_fixture,
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container._merge_timeseries()

    assert len(timeseries_container.timeseries) == 1
    assert timeseries_container.timeseries["time series merge"].measurements.to_list() == [2, 3, 4, 5, 6]
