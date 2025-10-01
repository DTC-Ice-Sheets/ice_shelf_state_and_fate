"""Module for creating a timeseries object."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import bootstrap
from shapely.geometry import MultiPolygon, Polygon
from sklearn.linear_model import LinearRegression

from use_case_3.raster_tools.raster_np import RasterNp
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    NormalisationType,
    TimeseriesRegion,
    TimeseriesType,
    UncertaintyType,
)


@dataclass
class Timeseries:
    """The timeseries data class. A wrapper to maintain metadata with the timeseries."""

    label: str
    timeseries_type: TimeseriesType
    timeseries_region: TimeseriesRegion
    measurement_name: str | None = None
    measurements: pd.Series | None = None
    measurement_std: pd.Series | None = None
    measurement_n: pd.Series | None = None
    measurement_uncertainties: pd.Series | None = None
    measurement_uncertainties_std: pd.Series | None = None
    measurement_uncertainties_n: pd.Series | None = None
    measurement_coverage: pd.Series | None = None
    measurement_coverage_n: pd.Series | None = None
    extent_polygon: Polygon | MultiPolygon | None = None
    unit: str = ""
    uncertainty: UncertaintyType = UncertaintyType.STD

    def process_timeseries(
        self: Timeseries,
        data: xr.Dataset,
        measurement_name: str,
        measurement_uncertainty_name: str | None,
        mask_geometries: gpd.GeoDataFrame,
        # mask_geometries: dict[str, Polygon | MultiPolygon | None],
        apply_mask: bool,
        aggregate: AggregationStrategy = AggregationStrategy.MEAN,
        period: AggregationPeriod | None = AggregationPeriod.MONTH,
        uncertainty: UncertaintyType | None = None,
        independent: bool = False,
        t0_normalisation: str | None = None,
    ) -> None:
        """Process the input data into a timeseries with associated uncertainties and coverage.

        This is the main method processing the datacube into the time series. This works with both 3D and 1D data.

        NOTE: Main entry point.
        NOTE: This function has been split into two separate functions depending on the data dimensionality. This is in
        attempt to simplify and make things more readable.

        Parameters
        ----------
        data : xr.Dataset
            The xarray dataset containing the measurement of interest as well as any other data of interest such as
            uncertainties. Ideally, this should be an xr.Dataset which has been read from an appropriately formatted
            GeoZarr.
        measurement_name : str
            The specific measurement to be processed. This must be a key of a DataArray with the data xr.Dataset.
        measurement_uncertainty_name: str | None
            The specific name of the uncertainty associated with the measurement. If None, it is assumed that the
            measurement will have no associated uncertainty.
        mask_geometries : dict[str, Polygon | MultiPolygon | None]
            The mask geometries. This is provided as a dictionary with three available masks. 1) The ice shelf, 2) the
            ocean and 3) The climate.
        apply_mask : bool
            Boolean flag whereby if True the mask supplied to the data will be used to mask the data. If False then the
            mask is only used to calculate the coverage.
        aggregate : AggregationStrategy
            The aggregation to perform on the data, by default AggregationStrategy.MEAN. Currently these are limited to:
            mean, median, max, min. Note, the aggregation chosen determines the spread according to the following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: None
        period : AggregationPeriod | None
            The time period over which to aggregate, by default AggregationPeriod.MONTH
        uncertainty : UncertaintyType | None
            The statistic type describing the uncertainty, by default None
        independent : bool
            Whether the uncertainties are independent or not, by default False
        t0_normalisation : str | None
            Whether to normalise the time series against time zero, by default None
        """
        data_observations = data[measurement_name].to_numpy()
        if measurement_uncertainty_name is not None:
            data_uncertainties = data[measurement_uncertainty_name].to_numpy()
            data_uncertainties[data_uncertainties < -200] = np.nan
        else:
            data_uncertainties = None
        dimension_data = {key: data[key] for key in data.coords}

        dimension_names = list(data[measurement_name].dims)
        if data.rio.crs is not None:
            mask_geometries = mask_geometries.to_crs(data.rio.crs)
        else:
            # currently only the SAM data has this issue, so this is a temporary fix
            mask_geometries = mask_geometries.to_crs("epsg:3031")

        # TODO: These will need to be better defined and formalised.
        if self.timeseries_region == TimeseriesRegion.ICE_SHELF:
            mask_geometry = mask_geometries.loc["ice_shelf", "geometry"]
        elif self.timeseries_region == TimeseriesRegion.OCEAN:
            mask_geometry = mask_geometries.loc["ocean", "geometry"]
        elif self.timeseries_region == TimeseriesRegion.CLIMATE:
            mask_geometry = mask_geometries.loc["climate", "geometry"]

        # If the input data has two or more dimensions, then it is necessary to aggregate, otherwise we assume it is a
        # 1d timeseries and simple add these to the timeseries object.
        # print(self.label, data_observations.shape)
        if len(data_observations.shape) > 1 and data_observations.shape[1] > 1 and data_observations.shape[2] > 1:
            self._process_timeseries_nd(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                dimension_data=dimension_data,
                dimension_names=dimension_names,
                mask_geometry=mask_geometry,
                measurement_uncertainty_name=measurement_uncertainty_name,
                aggregate=aggregate,
                period=period,
                uncertainty=uncertainty,
                independent=independent,
                t0_normalisation=t0_normalisation,
                apply_mask=apply_mask,
            )
        else:
            self._process_timeseries_1d(
                data_observations=data_observations,
                dimension_data=dimension_data,
                dimension_names=dimension_names,
                measurement_uncertainty_name=measurement_uncertainty_name,
                data_uncertainties=data_uncertainties,
                aggregate=aggregate,
                period=period,
                uncertainty=uncertainty,
                independent=independent,
                t0_normalisation=t0_normalisation,
            )
        self.uncertainty = uncertainty
        self.extent_polygon = mask_geometry

    def _store_processed_data(
        self: Timeseries,
        data_agg: pd.DataFrame,
        data_uncertainties_agg: pd.DataFrame | None = None,
        data_coverage_agg: pd.DataFrame | None = None,
    ) -> None:
        """
        Store the relvent data in the dataclass after processing.

        Parameters
        ----------
        data_agg : pd.DataFrame
            _The aggregated data.
        data_uncertainties_agg : pd.DataFrame | None, optional
            The aggregated data uncertainties, by default None
        data_coverage_agg : pd.DataFrame | None, optional
            The aggregated data coverage, by default None
        """
        self.measurements = data_agg["aggregation"]
        self.measurement_std = data_agg["data_std"]
        self.measurement_n = data_agg["aggregation_n"]

        if data_uncertainties_agg is not None:
            self.measurement_uncertainties = data_uncertainties_agg["uncertainty"]
            self.measurement_uncertainties_std = data_uncertainties_agg["uncertainty_std"]
            self.measurement_uncertainties_n = data_uncertainties_agg["uncertainty_n"]

        if data_coverage_agg is not None:
            self.measurement_coverage = data_coverage_agg["coverage"]
            self.measurement_coverage_n = data_coverage_agg["coverage_n"]

    def _process_timeseries_nd(
        self: Timeseries,
        data_observations: np.ndarray,
        dimension_data: dict[str, xr.DataArray],
        dimension_names: list[str],
        mask_geometry: Polygon | MultiPolygon | None = None,
        data_uncertainties: np.ndarray | None = None,
        measurement_uncertainty_name: str | None = None,
        aggregate: AggregationStrategy = AggregationStrategy.MEAN,
        period: AggregationPeriod | None = AggregationPeriod.MONTH,
        uncertainty: UncertaintyType | None = None,
        independent: bool = False,
        t0_normalisation: str | None = None,
        apply_mask: bool = False,
    ) -> None:
        """
        Process n-d data into a timeseries. Specifically this will be run when the number of dims > 1.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        dimension_data : dict[str, xr.DataArray]
            The dimension data.
        dimension_names : list[str]
            The dimension names.
        mask_geometry : Polygon | MultiPolygon | None
            The mask used, by default None
        data_uncertainties : np.ndarray | None
            The data uncertainties, by default None
        measurement_uncertainty_name : str | None
            The name of the uncertainties, by default None
        aggregate : AggregationStrategy
            The method of aggregation, by default AggregationStrategy.MEAN
        period : AggregationPeriod | None
            The period of aggregation, by default AggregationPeriod.MONTH
        uncertainty : UncertaintyType | None
            The type of the uncertainty such as standard deviation or standard error, by default None
        independent : bool
            Whether data points can be treated as being independent, by default False
        t0_normalisation : str | None
            Whether the data is normalised, by default None
        apply_mask : bool
            If true the data is masked, by default False
        """
        if mask_geometry is not None:
            resolution = float(np.abs(dimension_data["y"][1].to_numpy() - dimension_data["y"][0].to_numpy()))
            y_affine_negative = not dimension_data["y"][1].to_numpy() > dimension_data["y"][0].to_numpy()

            data_mask = RasterNp.rasterise_mask_from_geometry(
                geometries=[mask_geometry],
                y_min=float(dimension_data["y"].min().to_numpy()),
                y_max=float(dimension_data["y"].max().to_numpy() + resolution),
                x_min=float(dimension_data["x"].min().to_numpy()),
                x_max=float(dimension_data["x"].max().to_numpy() + resolution),
                resolution=resolution,
                y_affine_negative=y_affine_negative,
            )
        else:
            apply_mask = False

        if apply_mask:
            data_observations, modify = self._mask_data_cube(
                data_cube=data_observations, data_mask=data_mask, dimension_names=dimension_names, modify=True
            )
            if measurement_uncertainty_name is not None:
                data_uncertainties, modify = self._mask_data_cube(
                    data_cube=data_uncertainties,
                    data_mask=data_mask,
                    dimension_names=dimension_names,
                    modify=modify,
                )
        else:
            modify = True

        data_agg = self._calculate_aggregate_timeseries(
            data_observations=data_observations,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            aggregate=aggregate,
            period=period,
            t0_normalisation=t0_normalisation,
            modify=modify,
        )

        data_coverage_agg = self._calculate_coverage_timeseries(
            data_observations=data_observations,
            data_mask=data_mask,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            period=period,
        )

        if measurement_uncertainty_name is not None:
            data_uncertainties_agg = self._propagate_uncertainties_timeseries(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                dimension_data=dimension_data,
                dimension_names=dimension_names,
                aggregate=aggregate,
                uncertainty=uncertainty,
                period=period,
                independent=independent,
                t0_normalisation=t0_normalisation,
                modify=modify,
            )
        else:
            data_uncertainties_agg = None

        self._store_processed_data(
            data_agg=data_agg, data_uncertainties_agg=data_uncertainties_agg, data_coverage_agg=data_coverage_agg
        )

    def _process_timeseries_1d(
        self: Timeseries,
        data_observations: np.ndarray,
        dimension_data: dict[str, xr.DataArray],
        dimension_names: list[str],
        measurement_uncertainty_name: str | None = None,
        data_uncertainties: np.ndarray | None = None,
        aggregate: AggregationStrategy = AggregationStrategy.MEAN,
        period: AggregationPeriod | None = AggregationPeriod.MONTH,
        uncertainty: UncertaintyType | None = None,
        independent: bool = False,
        t0_normalisation: str | None = None,
    ) -> None:
        """
        Process 1-d data into a timeseries. Specifically this will be run when the number of dims = 1.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        dimension_data : dict[str, xr.DataArray]
            The dimension data.
        dimension_names : list[str]
            The dimension names.
        measurement_uncertainty_name : str | None
            The name of the uncertainties, by default None
        data_uncertainties : np.ndarray | None
            The data uncertainties, by default None
        aggregate : AggregationStrategy
            The method of aggregation, by default AggregationStrategy.MEAN
        period : AggregationPeriod | None
            The period of aggregation, by default AggregationPeriod.MONTH
        uncertainty : UncertaintyType | None
            The type of the uncertainty such as standard deviation or standard error, by default None
        independent : bool
            Whether data points can be treated as being independent, by default False
        t0_normalisation : str | None
            Whether the data is normalised, by default None
        """
        modify = True
        data_agg = self._calculate_aggregate_timeseries(
            data_observations=data_observations,
            dimension_data=dimension_data,
            dimension_names=dimension_names,
            aggregate=aggregate,
            period=period,
            t0_normalisation=t0_normalisation,
            modify=modify,
        )
        data_agg = data_agg.loc[~data_agg.index.duplicated(keep="first")]
        self.measurement_coverage = None
        self.measurement_coverage_n = None
        if measurement_uncertainty_name is not None:
            data_uncertainties_agg = self._propagate_uncertainties_timeseries(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                dimension_data=dimension_data,
                dimension_names=dimension_names,
                aggregate=aggregate,
                uncertainty=uncertainty,
                period=period,
                independent=independent,
                t0_normalisation=t0_normalisation,
                modify=modify,
            )
            data_uncertainties_agg = data_uncertainties_agg.loc[~data_uncertainties_agg.index.duplicated(keep="first")]
        else:
            data_uncertainties_agg = None

        self._store_processed_data(
            data_agg=data_agg, data_uncertainties_agg=data_uncertainties_agg, data_coverage_agg=None
        )

    def _calculate_aggregate_timeseries(
        self: Timeseries,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: list[str],
        aggregate: AggregationStrategy = AggregationStrategy.MEAN,
        period: AggregationPeriod | None = None,
        t0_normalisation: str | None = None,
        modify: bool = False,
    ) -> pd.DataFrame:
        """
        Aggregate 2D or 3D data variable into a 1D time series using a specified aggregation.

        A 3D dataset will result in a 1D timeseries, while a 2D dataset will result in a single point in time.

        Parameters
        ----------
        data_observations : np.ndarray
            Input data with axes ordered according to the dimension names.
        dimension_data : dict
            The data for each of the data dimensions, i.e. coordinate to_numpy().
        dimension_names : list[str]
            List of dimension names with indexes corresponding to the axes in data.
        aggregate : AggregationStrategy
            The aggregation to perform on the data, by default "mean". Currently these are limited to:
            mean, median, max, min. Note, the aggregation chosen determines the spread according to the following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: None
        period : AggregationPeriod | None
            The time period over which to aggregate, by default None.
        t0_normalisation : str | None
            Whether to normalise the time series against time zero, by default None.
        modify : bool
            Whether to modify the dimensions of the datacube, by default False.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the aggregated variable, the spread of that variable and the number of
            points in the aggregation. The index of the dataframe is time.
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=modify
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        aggregate_spread, count_n = self._spread_calculation_type(aggregate=aggregate)

        time_idx = self._process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        data_agg, time_agg = self._aggregate_statistic(
            data=data_observations, dimension_names=dimension_names, aggregate=aggregate, time_index=time_idx
        )

        if t0_normalisation is not None:
            data_agg = self.t0_normalisation(data_observation=data_agg, normalisation_type=t0_normalisation)

        if aggregate_spread != AggregationStrategy.NONE:
            data_agg_spread, _ = self._aggregate_statistic(
                data=data_observations, dimension_names=dimension_names, aggregate=aggregate_spread, time_index=time_idx
            )
        else:
            data_agg_spread = np.zeros(data_agg.shape[0])

        if count_n:
            data_agg_n = self._count_nan_elements(data=data_observations, non_nan=True, time_index=time_idx)
        else:
            data_agg_n = np.zeros(data_agg.shape[0])

        return pd.DataFrame(
            np.hstack((data_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))),
            columns=["aggregation", "data_" + aggregate_spread.value, "aggregation_n"],
            index=time_agg,
        )

    def _propagate_uncertainties_timeseries(
        self: Timeseries,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        dimension_data: dict,
        dimension_names: list[str],
        aggregate: AggregationStrategy = AggregationStrategy.MEAN,
        uncertainty: UncertaintyType = UncertaintyType.STD,
        period: AggregationPeriod | None = None,
        independent: bool = False,
        t0_normalisation: str | None = None,
        modify: bool = False,
    ) -> pd.DataFrame:
        """
        Propagate the uncertainties.

        A 3D dataset will result in a 1D timeseries, while a 2D dataset will result in a single point in time.

        Parameters
        ----------
        data_observations : np.ndarray
            Input observation data with axes ordered according to the dimension names.
        data_uncertainties : np.ndarray
            Input data uncertainties with axes ordered according to the dimension names.
        dimension_data : dict
            The data for each of the data dimensions, i.e. coordinate to_numpy().
        dimension_names : list[str]
            List of dimension names with indexes corresponding to the axes in data.
        aggregate : AggregationStrategy
            The type of aggregation performed on the data, by default AggregationStrategy.MEAN. Currently these are
            limited to: mean, median, max, min. Note, the aggregation chosen determines the spread according to the
            following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: NaN
        uncertainty : UncertaintyType
            The type of uncertainty provided, by default UncertaintyType.STD. Currently these are limited to:
                - std
        period : AggregationPeriod | None
            The time period over which to aggregate, by default None.
        independent : bool
            Whether the uncertainties are independent or not, by default False
        t0_normalisation : str | None
            Whether to normalise the time series against time zero, by default None.
        modify : bool
            Whether to modify the dimensions of the datacube, by default False.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the aggregated uncertainties, the spread of that uncertainties and the number of
            points in the aggregation. The index of the dataframe is time.
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=modify
        )
        data_uncertainties, dimension_names = self._expand_dimensions(
            data=data_uncertainties, dimension_names=dimension_names, modify_dimension_names=modify
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        data_uncertainties = self._move_axes_time(data=data_uncertainties, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        time_idx = self._process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        if aggregate == AggregationStrategy.MEAN:
            data_uncertainties_agg, time_agg = self._aggregate_uncertainties_mean(
                data_uncertainties=data_uncertainties,
                uncertainty=uncertainty,
                time_index=time_idx,
                independent=independent,
            )
        elif aggregate == AggregationStrategy.MEDIAN:
            data_uncertainties_agg, time_agg = self._aggregate_uncertainties_median(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                time_index=time_idx,
                n_repeats=100,
            )
        elif aggregate == AggregationStrategy.MAX or aggregate == AggregationStrategy.MIN:
            data_uncertainties_agg, time_agg = self._aggregate_uncertainties_minmax(
                data_observations=data_observations,
                data_uncertainties=data_uncertainties,
                uncertainty=uncertainty,
                time_index=time_idx,
            )

        if t0_normalisation is not None:
            self._t0_normalisation_uncertainties(data_uncertainties_agg, normalisation_type=t0_normalisation)

        data_agg_n = self._count_nan_elements(data=data_uncertainties, non_nan=True, time_index=time_idx)

        data_agg_spread, _ = self._aggregate_statistic(
            data=data_uncertainties,
            dimension_names=dimension_names,
            aggregate=AggregationStrategy.STD,
            time_index=time_idx,
        )

        return pd.DataFrame(
            np.hstack(
                (data_uncertainties_agg.reshape((-1, 1)), data_agg_spread.reshape((-1, 1)), data_agg_n.reshape((-1, 1)))
            ),
            columns=["uncertainty", "uncertainty_std", "uncertainty_n"],
            index=time_agg,
        )

    @staticmethod
    def _count_nan_elements(data: np.ndarray, non_nan: bool = False, time_index: dict | None = None) -> np.ndarray:
        """
        Count the number of nan or non-nan elements in an array.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        non_nan : bool
            If False then the number of NaN to_numpy() are counted, if True then the number of non-NaN to_numpy() are
            counted, by default False.
        time_index : dict | None
            The time indexes over which to do the aggregations, by default None

        Returns
        -------
        np.ndarray
            The count of either NaN or non-NaN to_numpy().
        """
        if time_index is None:
            time_index = {0: [0]}

        data_agg_n = []
        for time_point in time_index:
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            if non_nan:
                data_agg_n.append(np.count_nonzero(~np.isnan(data[time_min:time_max, :, :])))
            else:
                data_agg_n.append(np.count_nonzero(np.isnan(data[time_min:time_max, :, :])))
        return np.array(data_agg_n)

    @staticmethod
    def _aggregate_statistic(
        data: np.ndarray,
        aggregate: AggregationStrategy,
        time_index: dict | None = None,
        dimension_names: list[str] | None = None,
        dimension_aggregate: int | tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list]:
        """
        Calculate the mean along the none time axes resulting in a 1d timeseries of mean to_numpy() with associated
        uncertainties.

        Parameters
        ----------
        data : np.ndarray
            The input n-dim data array. When the array is 1d along the time axis this will be returned with no
            aggregation.
        aggregate : AggregationStrategy
            The type of aggregation performed on the data. Currently these are limited to:
            mean, median, max, min. Note, the aggregation chosen determines the spread according to the following rules:
                - mean: standard deviation
                - median: IQR
                - max/min: NaN
        time_index : dict | None
            The time indexes over which to do the aggregations, by default None
        dimension_names : list[str] | None
            Specification of the dimension names.
        dimension_aggregate : int | tuple[int, int] | None
            The dimension of the data array to aggregate against, by default None.

        Returns
        -------
        tuple[np.ndarray, list]
            The aggregated data.
        """  # noqa: D205
        # If the dimensions over which to aggregate are not specified, then infer this from the dim_names:
        if dimension_aggregate is None:
            dimension_aggregate = tuple(i for i, dim_name in enumerate(dimension_names) if dim_name != "time")

        time_coords_agg = []
        data_agg = []

        if time_index is None:
            time_index = {0: [0]}

        for time_point in time_index:
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            if aggregate == AggregationStrategy.MEAN:
                data_agg.append(np.nanmean(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == AggregationStrategy.MEDIAN:
                data_agg.append(np.nanmedian(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == AggregationStrategy.MAX:
                data_agg.append(np.nanmax(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == AggregationStrategy.MIN:
                data_agg.append(np.nanmin(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == AggregationStrategy.STD:
                data_agg.append(np.nanstd(data[time_min:time_max, :, :]))
                time_coords_agg.append(time_point)
            elif aggregate == AggregationStrategy.IQR:
                data_agg.append(np.subtract(*np.nanpercentile(data[time_min:time_max, :, :], q=[75, 25])))
                time_coords_agg.append(time_point)

        return np.array(data_agg), time_coords_agg

    def _aggregate_uncertainties_mean(
        self: Timeseries,
        data_uncertainties: np.ndarray,
        uncertainty: UncertaintyType,
        time_index: dict | None = None,
        independent: bool = False,
    ) -> tuple[np.ndarray, list]:
        """
        Aggregate the uncertainties when aggregating using the mean.

        In this case the standard deviations are combined through quadrature.

        Parameters
        ----------
        data_uncertainties : np.ndarray
            Input data uncertainties with axes ordered according to the dimension names.
        uncertainty : UncertaintyType
            The type of uncertainty, currently only std is supported.
        time_index : dict | None
            The time indexes over which to do the aggregations, by default None
        independent : bool
            Whether the errors are treated as independent when propagating. If they are assumed to be
            independent then they are propagated using quatrature, if they are not independent, then the errors are
            summed as a worse case.

        Returns
        -------
        tuple[np.ndarray, list]
            The aggregated uncertainties and a list of time indexes.
        """
        data_agg_n = self._count_nan_elements(data=data_uncertainties, non_nan=True, time_index=time_index)

        time_coords_agg = []
        data_agg = []

        if time_index is None:
            time_index = {0: [0]}

        for i, time_point in enumerate(time_index):
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            if uncertainty == UncertaintyType.STD:
                if independent:
                    data_agg.append(
                        np.sqrt(np.nansum(np.square(data_uncertainties[time_min:time_max, :, :]))) / data_agg_n[i]
                    )
                else:
                    data_propagated_space = np.zeros((time_max - time_min))
                    for time_idx in range(time_min, time_max):
                        data_agg_n_time_point = np.count_nonzero(~np.isnan(data_uncertainties[time_idx, :, :]))
                        data_propagated_space[time_idx - time_min] = (
                            np.nansum(data_uncertainties[time_min:time_max, :, :]) / data_agg_n_time_point
                        )
                    data_agg.append(np.sqrt(np.nansum(np.square(data_propagated_space))) / (time_max - time_min))

        return np.array(data_agg), time_coords_agg

    def _aggregate_uncertainties_median(
        self: Timeseries,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        time_index: dict | None = None,
        n_repeats: int = 1000,
    ) -> tuple[np.ndarray, list]:
        """
        Aggregate the uncertainties for the median assuming independence.

        This takes a sample of the original data and then uses bootstrapping to estimate the propagation of
        the uncertainty.

        Parameters
        ----------
        data_observations : np.ndarray
            The observation data.
        data_uncertainties : np.ndarray
            The uncertainty data.
        time_index : dict | None
            The time indexes over which to do the aggregations, by default None
        n_repeats : int
            The number of repeats. This is both the number of samples drawn and the number of bootstrapping iterations,
            by default 1000

        Returns
        -------
        tuple[np.ndarray, list]
            The aggregated uncertainties and a list of time indexes.
        """
        time_coords_agg = []
        data_agg = []

        if time_index is None:
            time_index = {0: [0]}

        for time_point in time_index:
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_observation_time_point = data_observations[time_min:time_max, :, :].flatten()
            data_uncertainties_time_point = data_uncertainties[time_min:time_max, :, :].flatten()

            data_uncertainties_time_point = data_uncertainties_time_point[~np.isnan(data_observation_time_point)]
            data_observation_time_point = data_observation_time_point[~np.isnan(data_observation_time_point)]

            # Resample the data
            resampled_data = np.random.normal(
                loc=data_observation_time_point[0],
                scale=data_uncertainties_time_point[0],
                size=(data_uncertainties_time_point.shape[0], n_repeats),
            )

            # Use bootstrapping to estimate the uncertainties
            res = bootstrap((resampled_data,), np.median, confidence_level=0.95, n_resamples=n_repeats)
            data_agg.append(np.mean(res.standard_error))

        return np.array(data_agg), time_coords_agg

    def _aggregate_uncertainties_minmax(
        self: Timeseries,
        data_observations: np.ndarray,
        data_uncertainties: np.ndarray,
        uncertainty: UncertaintyType,
        time_index: dict | None = None,
    ) -> tuple[np.ndarray, list]:
        """
        Aggregate the uncertainties when aggregating using either the maximum or minimum value.

        In this case the standard deviation corresponds to the pixel with either the maximum or minimum
        value respectively.

        Parameters
        ----------
        data_observations : np.ndarray
            Input data with axes ordered according to the dimension names.
        data_uncertainties : np.ndarray
            Input data uncertainties with axes ordered according to the dimension names.
        uncertainty : UncertaintyType
            The type of uncertainty, currently only std is supported.
        time_index : dict | None
            The time indexes over which to do the aggregations, by default None

        Returns
        -------
        tuple[np.ndarray, list]
            The aggregated uncertainties and a list of time indexes.
        """
        time_coords_agg = []
        data_agg = []

        if time_index is None:
            time_index = {0: [0]}

        for time_point in time_index:
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_subcube_observations = data_observations[time_min:time_max, :, :]
            data_subcube_uncertainties = data_uncertainties[time_min:time_max, :, :]
            if uncertainty == UncertaintyType.MAX:
                idx = np.unravel_index(
                    np.nanargmax(data_subcube_observations, axis=None), data_subcube_observations.shape
                )
                data_agg.append(data_subcube_uncertainties[idx])
            elif uncertainty == UncertaintyType.MIN:
                idx = np.unravel_index(
                    np.nanargmin(data_subcube_observations, axis=None), data_subcube_observations.shape
                )
                data_agg.append(data_subcube_uncertainties[idx])

        return np.array(data_agg), time_coords_agg

    def _t0_normalisation(
        self: Timeseries,
        data_observation: np.ndarray,
        normalisation_type: NormalisationType = NormalisationType.FIRST,
    ) -> np.ndarray | None:
        """
        Normalise a 1D time series against a time zero (t0).

        Parameters
        ----------
        data_observation : np.ndarray
            The 1D timeseries to be normalised.
        normalisation_type : NormalisationType
            The normalisation type to be applied. Using NormalisationType.FIRST subtracts
            the first value from the whole time series, whereas using NormalisationType.LINEAR_REGRESSION fits a
            unregularised linear regression model to to the whole time series and then uses a predicted t0 to normalise
            the dataset, by default NormalisationType.FIRST.

        Returns
        -------
        np.ndarray | None
            The normalised 1D timeseries using the chosen normalisation method.
        """
        if normalisation_type == NormalisationType.FIRST:
            return data_observation - data_observation[0]
        if normalisation_type == NormalisationType.LINEAR_REGRESSION:
            x = np.arange(data_observation.shape[0])
            model = LinearRegression().fit(x.reshape((-1, 1)), data_observation.reshape((-1, 1)))
            y_0 = model.predict(np.array(0).reshape((-1, 1)))
            return data_observation - y_0
        return None

    def _t0_normalisation_uncertainties(
        self: Timeseries,
        data_uncertainties: np.ndarray | None,
        normalisation_type: NormalisationType = NormalisationType.FIRST,
    ) -> np.ndarray | None:
        """
        Propagate uncertainties when normalising a 1D time series against a time zero (t0).

        Parameters
        ----------
        data_uncertainties : np.ndarray | None
            The 1D timeseries to be normalised.
        normalisation_type : NormalisationType
            The normalisation type to be applied. Using NormalisationType.FIRST subtracts
            the first value from the whole time series, whereas using NormalisationType.LINEAR_REGRESSION fits a
            unregularised linear regression model to to the whole time series and then uses a predicted t0 to normalise
            the dataset, by default NormalisationType.FIRST.

        Returns
        -------
        np.ndarray | None
            The normalised 1D timeseries using the chosen normalisation method.
        """
        if normalisation_type == NormalisationType.FIRST:
            return self._combine_uncertainties_quadrature(data_uncertainties, data_uncertainties[0])
        if normalisation_type == NormalisationType.LINEAR_REGRESSION:
            # This needs fixing as the uncertainties will not be propagated correctly when fitting the model.
            return self._combine_uncertainties_quadrature(data_uncertainties, data_uncertainties[0])
        return None

    def _calculate_percentiles_timeseries(
        self: Timeseries,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: list[str],
        period: AggregationPeriod | None = None,
        percentiles: tuple[float, float] = (25, 75),
    ) -> pd.DataFrame:
        """
        Calculate the upper and lower percentiles of a datacube of observations.

        Parameters
        ----------
        data_observations : np.ndarray
            The datacube of observations.
        dimension_data : dict
            The coordinate data.
        dimension_names : list[str]
            The dimension names
        period : AggregationPeriod | None
            The period over which the data is aggregated, by default None
        percentiles : tuple[float, float]
            The lower and upper percentile to be calculated for a datacube. These must be a tuple of floats in the range
            [0.0, 100.0], by default (25, 75)

        Returns
        -------
        pd.DataFrame
            A dataframe of the upper/lower percentiles and the number of data points considered in each time slice. The
            index of the dataframe is the first time point for a given time slice.
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=True
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        time_idx = self._process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        data_percentile_lower, time_agg = self._determine_percentiles(
            data_observations=data_observations, time_index=time_idx, percentile=percentiles[0]
        )

        data_percentile_upper, _ = self._determine_percentiles(
            data_observations=data_observations, time_index=time_idx, percentile=percentiles[1]
        )

        data_percentile_n = self._count_nan_elements(data=data_observations, non_nan=True, time_index=time_idx)

        return pd.DataFrame(
            np.hstack(
                (
                    data_percentile_lower.reshape((-1, 1)),
                    data_percentile_upper.reshape((-1, 1)),
                    data_percentile_n.reshape((-1, 1)),
                )
            ),
            columns=[f"percentile_{percentiles[0]}", f"percentile_{percentiles[1]}", "percentile_n"],
            index=time_agg,
        )

    def _determine_percentiles(
        self: Timeseries, data_observations: np.ndarray, time_index: dict | None, percentile: float
    ) -> tuple[np.ndarray, list]:
        """
        Calculate the percentile for a given datacube and time index.

        Parameters
        ----------
        data_observations : np.ndarray
            The data cube of observations.
        time_index : dict | None
            The time indexes for each time slice to be processed.
        percentile : float
            The percentile to be calculated. This needs to be a float in the range [0.0, 100.0] inclusive.

        Returns
        -------
        tuple[np.ndarray, list]
            The percentiles and associated time coordinates for each percentile.
        """
        data_percentile = []
        time_coords_agg = []

        if time_index is None:
            time_index = {0: [0]}

        for time_point in time_index:
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_percentile.append(np.nanpercentile(data_observations[time_min:time_max, :, :], percentile))
        return np.array(data_percentile), time_coords_agg

    def _calculate_coverage_timeseries(
        self: Timeseries,
        data_observations: np.ndarray,
        dimension_data: dict,
        dimension_names: list[str],
        data_mask: np.ndarray | None = None,
        period: AggregationPeriod | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the coverage for the timeseries.

        Coverage is calculated against a reference mask.
        If no mask is provided the coverage is calculated against the total pixels.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        dimension_data : dict
            The coordinate data.
        dimension_names : list[str]
            The dimension names
        data_mask : np.ndarray | None
            The data mask, by default None
        period : AggregationPeriod | None
            The period over which the data is aggregated, by default None

        Returns
        -------
        pd.DataFrame
            The time series of the coverage.
        """
        data_observations, dimension_names = self._expand_dimensions(
            data=data_observations, dimension_names=dimension_names, modify_dimension_names=True
        )

        data_observations = self._move_axes_time(data=data_observations, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        time_idx = self._process_time_indexes(
            dimension_names=dimension_names, dimension_data=dimension_data, period=period
        )

        data_coverage, time_agg = self._determine_coverage(
            data_observations=data_observations, mask_raster=data_mask, time_index=time_idx
        )

        data_coverage_n = self._count_nan_elements(data=data_observations, non_nan=True, time_index=time_idx)

        return pd.DataFrame(
            np.hstack((data_coverage.reshape((-1, 1)) * 100, data_coverage_n.reshape((-1, 1)))),
            columns=["coverage", "coverage_n"],
            index=time_agg,
        )

    def _determine_coverage(
        self, data_observations: np.ndarray, time_index: dict | None, mask_raster: np.ndarray
    ) -> tuple[np.ndarray, list]:
        """
        Calculate the spatial coverage for each time slice.

        The spatial coverage is defined as the number of pixels with with a numeric value in the data-cube
        divided by the number of pixels in the mask over the same number of time points.
        If no mask is provided the coverage is calculated against the total pixels.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        time_index : dict | None
            The time indexes for each time slice to be processed.
        mask_raster : np.ndarray
            The binary mask used on the dataset.

        Returns
        -------
        tuple[np.ndarray, list]
            The spacial coverage for each time slice and the time index.
        """
        data_coverage = []
        time_coords_agg = []

        if time_index is None:
            time_index = {0: [0]}

        for time_point in time_index:
            time_min = time_index[time_point][0]
            time_max = time_index[time_point][-1] + 1
            time_coords_agg.append(time_point)
            data_coverage.append(
                self._determine_coverage_point(
                    data_observations=data_observations[time_min:time_max, :, :], mask_raster=mask_raster
                )
            )
        return np.array(data_coverage), time_coords_agg

    def _determine_coverage_point(self: Timeseries, data_observations: np.ndarray, mask_raster: np.ndarray) -> float:
        """
        Calculate the coverage for a specific data cube.

        If no mask is provided the coverage is calculated against the total pixels.

        Parameters
        ----------
        data_observations : np.ndarray
            The data observations.
        mask_raster : np.ndarray
            The rasterised mask.

        Returns
        -------
        float
            The spatial coverage.
        """
        n_observations = self._count_pixels(data=data_observations)
        # Multiply by the number of time points.
        n_raster = self._count_pixels(data=mask_raster) * data_observations.shape[0]

        return n_observations / n_raster

    @staticmethod
    def _count_pixels(data: np.ndarray) -> float | int:
        """
        Count the number of valid pixels.

        In the case of a boolean mask this is the number of False elements. In all other cases this is the number of
        non-NaN elements. If no mask is provided the coverage is calculated against the total pixels.

        Parameters
        ----------
        data : np.ndarray
            The data cube.

        Returns
        -------
        float | int
            The number of elements which are False or non-NaN.
        """
        # Assumes that False in mask are included pixels.
        if data.dtype == bool:
            return np.count_nonzero(~data)
        return np.count_nonzero(~np.isnan(data))

    @staticmethod
    def _combine_uncertainties_quadrature(
        data_uncertainties_1: np.float64 | np.ndarray, data_uncertainties_2: np.float64 | np.ndarray
    ) -> np.float64 | np.ndarray:
        """
        Combine uncertainties through quadrature.

        Parameters
        ----------
        data_uncertainties_1 : np.float64 | np.ndarray
            The first vector of uncertainties.
        data_uncertainties_2 : np.float64 | np.ndarray
            The second vector of uncertainties.

        Returns
        -------
        np.float64 | np.ndarray
            The combined uncertainty vector.
        """
        return np.sqrt(np.square(data_uncertainties_1) + np.square(data_uncertainties_2))

    @staticmethod
    def _move_axes_time(data: np.ndarray, dimension_names: list[str]) -> np.ndarray:
        """
        Reorder the axis so that the time axis is the first axis.

        If time is not in dimension names, then it returns the same data array and associated dimension names.

        Parameters
        ----------
        data : np.ndarray
            Input data with axes ordered according to the dimension names.
        dimension_names : list[str]
            List of dimension names with indexes corresponding to the axes in data.

        Returns
        -------
        np.ndarray
            The reordered data.
        """
        dimension_aggregate = dimension_names.index("time") if "time" in dimension_names else 0

        return np.moveaxis(data, dimension_aggregate, 0) if dimension_aggregate != 0 else data

    @staticmethod
    def _move_axes_time_labels(dimension_names: list[str]) -> list[str]:
        """
        Reorder the axis so that the time axis is the first axis.

        If time is not in dimension names, then it returns the same data array and associated dimension names.
        This only operates on the labels and not the array.

        Parameters
        ----------
        dimension_names : list[str]
            List of dimension names with indexes corresponding to the axes in data.

        Returns
        -------
        list[str]
            The reordered data and dimension names.
        """
        dimension_aggregate = dimension_names.index("time") if "time" in dimension_names else 0

        if dimension_aggregate != 0:
            dimension_names_reordered = [dimension_names.pop(dimension_aggregate), *dimension_names]
        else:
            dimension_names_reordered = dimension_names.copy()

        return dimension_names_reordered

    def _expand_dimensions(
        self: Timeseries, data: np.ndarray, dimension_names: list[str], modify_dimension_names: bool = False
    ) -> tuple[np.ndarray, list]:
        """
        Check the dimensions of the input array and if not 3D then add dummy dimensions to make the data 3D.

        The data array is padded with dummy dimensions, while the dimension names have the name '_DUMMY' added at
        the correct axis.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        dimension_names : list[str]
            The input dimension names.
        modify_dimension_names : bool, optional
            Whether the dimension_names should be modified, this is used when multiple data arrays with the same
            dimensions need to be modified, by default False.

        Returns
        -------
        tuple[np.ndarray, list]
            A 3D data array with three named dimensions.
        """
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)

        if modify_dimension_names:
            if len(dimension_names) == 2:
                dimension_names = ["blank", *dimension_names]
                # dimension_names = ["blank"] + dimension_names
            elif len(dimension_names) == 1:
                dimension_names = ["blank"] * 2 + dimension_names

        return data, dimension_names

    def _find_time_indexes(self: Timeseries, time_coord: pd.Series, period: AggregationPeriod) -> dict:
        """
        Find the time indexes to slice into the data-cube.

        Parameters
        ----------
        time_coord : pd.Series
            The time coordinates.
        period : AggregationPeriod
            The period over which to aggregate.

        Returns
        -------
        dict
            Returns the dictionary, mapping each time slice to the data-cube indexes.
        """
        if isinstance(time_coord.iloc[0], str):
            time_coord = time_coord.str.replace("-", "/")
            time_coord = time_coord.str.replace("_", "/")
        time_coord_df = time_coord.to_frame().reset_index(drop=True)
        time_coord_df["time"] = pd.to_datetime(time_coord_df["time"], format="%Y/%m/%d")
        time_coord_df["year"] = time_coord_df["time"].dt.year.astype(int)
        time_coord_df["month"] = time_coord_df["time"].dt.month.astype(int)

        time_coord_df["date_month"] = pd.to_datetime(
            [f"{y}-{m}" for y, m in zip(time_coord_df["year"], time_coord_df["month"], strict=False)]
        )

        if period == AggregationPeriod.YEAR:
            return {year: time_coord_df[time_coord_df["year"] == year].index for year in time_coord_df["year"].unique()}
        if period == AggregationPeriod.MONTH:
            return {
                date_month: time_coord_df[time_coord_df["date_month"] == date_month].index
                for date_month in time_coord_df["date_month"].unique()
            }
        return None

    def _process_time_indexes(
        self: Timeseries,
        dimension_names: list[str],
        dimension_data: xr.DataArray | pd.DataFrame | dict,
        period: AggregationPeriod | None,
    ) -> dict:
        """
        Process the time indexes.

        This returns a dictionary with the each key containing the indexes to be aggregated into a single point.

        Parameters
        ----------
        dimension_names : list[str]
            The list of dimension names.
        dimension_data : xr.DataArray | pd.DataFrame | dict
            The coordinate data, i.e. the grid coordinates and the time coordinates.
        period : AggregationPeriod | None
            The time period over which to aggregate the data points. If no periodis chosen then the dictionary will
            contain one time point per key.

        Returns
        -------
        dict
            The dictionary containing the indexes for the time points to be aggregated.
        """
        if "time" in dimension_names:
            if isinstance(dimension_data, xr.DataArray) and "time":
                time_coord = dimension_data["time"].to_pandas()
            elif isinstance(dimension_data, pd.DataFrame):
                time_coord = dimension_data["time"]
            elif isinstance(dimension_data, dict):
                time_coord = pd.Series(dimension_data["time"], name="time")

            if period is not None:
                return self._find_time_indexes(time_coord=time_coord, period=period)
            return {datetime: [idx] for idx, datetime in time_coord.items()}
        return {0: [0]}

    @staticmethod
    def _spread_calculation_type(aggregate: AggregationStrategy) -> tuple[AggregationStrategy | None, bool]:
        """
        Specify parameters which are calculated for the spread based on the aggregate function applied.

        Parameters
        ----------
        aggregate : AggregationStrategy
            The type of aggregation applied to calculate the central tendency.

        Returns
        -------
        tuple[AggregationStrategy | None, bool]
            The aggregate function for the spread, and the counts.
        """
        if aggregate == AggregationStrategy.MEAN:
            aggregate_spread = AggregationStrategy.STD
            count_n = True
        elif aggregate == AggregationStrategy.MEDIAN:
            aggregate_spread = AggregationStrategy.IQR
            count_n = True
        elif aggregate in [AggregationStrategy.MAX, AggregationStrategy.MIN]:
            aggregate_spread = AggregationStrategy.NONE
            count_n = False

        return aggregate_spread, count_n

    def _mask_data_cube(
        self: Timeseries,
        data_cube: np.ndarray,
        data_mask: np.ndarray,
        dimension_names: list[str],
        modify: bool,
    ) -> tuple[np.ndarray, bool]:
        """
        Mask a data cube based on a given mask.

        This makes some strong assumptions on the data and mask:
            - 1) The data mask and the data are both in the same projection.
            - 2) The data mask and the data are both either y_affine_negative or y_affine_positive.
            - 3) The data cube will be rearranged so that the time axis is the first axis, however no checks will be
                 done that the second and third axis will be the same as the mask. There is a check on the dimensions,
                 but in the case of the mask being square, the dimensions could be wrong.

        Parameters
        ----------
        data_cube : np.ndarray
            The data cube to be masked.
        data_mask : np.ndarray
            The rasterised data mask.
        dimension_names : list[str]
            A list of the dimension names. These need to be in the order that they index the axes of the datacube.
        modify : bool
            Whether to modify the dimensions of the datacube.

        Returns
        -------
        tuple[np.ndarray, bool]
            The masked data cube.

        Raises
        ------
        ValueError
            The dimensions of the mask and the second and third axis of the datacube don't match.
        """
        data_cube, dimension_names = self._expand_dimensions(
            data=data_cube, dimension_names=dimension_names, modify_dimension_names=modify
        )

        data_cube = self._move_axes_time(data=data_cube, dimension_names=dimension_names)
        dimension_names = self._move_axes_time_labels(dimension_names=dimension_names)

        nan_array = np.empty((data_cube.shape[1], data_cube.shape[2]))
        nan_array[:] = np.nan

        if data_mask.shape != (data_cube.shape[1], data_cube.shape[2]):
            raise ValueError(
                f"Your data {data_cube.shape[1], data_cube.shape[2]} and mask {data_mask.shape} shapes are different"
            )

        data_cube_masked = np.where(data_mask, nan_array, data_cube)

        if modify:
            modify = False

        return data_cube_masked, modify

    def plot(
        self: Timeseries,
        spread: str | None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> None:
        """
        Plot the timeseries data.

        Parameters
        ----------
        spread : str | None
            The spread to display on the line plot. This can be one of None, "uncertainties" and "percentiles".
        x_label : str | None
            A X-axis label to use, by default None
        y_label : str | None
            A Y-axis label to use, by default None
        """
        error_pos = None
        error_neg = None

        if spread == "uncertainties" and self.measurement_uncertainties is not None:
            error_pos = self.measurements + self.measurement_uncertainties
            error_neg = self.measurements - self.measurement_uncertainties

        _fig, axes = plt.subplots(1, 1)
        sns.lineplot(
            x=self.measurements.index,
            y=self.measurements,
            color="black",
            ax=axes,
            linewidth=0.5,
        )
        sns.scatterplot(
            x=self.measurements.index,
            y=self.measurements,
            color="black",
            ax=axes,
            s=10,
            linewidth=0.5,
        )

        if spread == "uncertainties" and self.measurement_uncertainties is not None:
            sns.lineplot(
                x=self.measurements.index,
                y=error_pos,
                color="black",
                alpha=0.5,
                linewidth=0.5,
                linestyle="--",
                ax=axes,
            )

            sns.lineplot(
                x=self.measurements.index,
                y=error_neg,
                color="black",
                alpha=0.5,
                linewidth=0.5,
                linestyle="--",
                ax=axes,
            )
            plt.fill_between(
                self.measurements.index,
                error_pos,
                error_neg,
                color="black",
                alpha=0.25,
            )
        plt.grid()
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        else:
            plt.ylabel(f"{self.measurement_name.replace('_', ' ').title()} ({self.unit})")
        plt.show()

    def plot_coverage(
        self: Timeseries,
        x_label: str | None = None,
    ) -> None:
        """
        Plot the data coverage.

        The data coverage is defined as the percentage of points within a masked area with a valid measurement.

        Parameters
        ----------
        x_label : str | None
            A X-axis label to use, by default None
        """
        _fig, axes = plt.subplots(1, 1)
        sns.lineplot(
            x=self.measurement_coverage.index,
            y=self.measurement_coverage,
            color="black",
            ax=axes,
            linewidth=0.5,
        )
        sns.scatterplot(
            x=self.measurement_coverage.index,
            y=self.measurement_coverage,
            color="black",
            ax=axes,
            s=10,
            linewidth=0.5,
        )
        plt.grid()
        if x_label is not None:
            plt.xlabel(x_label)
        plt.ylabel("Coverage (%)")
        plt.show()

    def summary_statistics(self: Timeseries) -> None:
        """
        Generate a set of summary statistics for a time series.

        This assumes that the timeseries is ordered. It also assumes that there are associated uncertainties.
        """
        start = self.measurements.iloc[0]
        start_uncertainty = self.measurement_uncertainties.iloc[0]
        end = self.measurements.iloc[-1]
        end_uncertainty = self.measurement_uncertainties.iloc[-1]
        difference = end - start
        difference_uncertainty = self._combine_uncertainties_quadrature(start_uncertainty, end_uncertainty)

        value_maximum = self.measurements.max()
        value_maximum_uncertainty = self.measurement_uncertainties[self.measurements.idxmax()]

        value_minimum = self.measurements.min()
        value_minimum_uncertainty = self.measurement_uncertainties[self.measurements.idxmin()]
        percentile_90 = np.percentile(self.measurements, 90)
        percentile_75 = np.percentile(self.measurements, 75)
        percentile_50 = np.percentile(self.measurements, 50)
        percentile_25 = np.percentile(self.measurements, 25)
        percentile_10 = np.percentile(self.measurements, 10)
        if self.measurement_coverage is not None:
            coverage_maximum = self.measurement_coverage.max()
            coverage_minimum = self.measurement_coverage.min()
            coverage_range = coverage_maximum - coverage_minimum
            coverage_mean = self.measurement_coverage.mean()
            coverage_std = self.measurement_coverage.std()
            coverage_median = self.measurement_coverage.median()
        else:
            coverage_maximum = np.nan
            coverage_minimum = np.nan
            coverage_range = np.nan
            coverage_mean = np.nan
            coverage_std = np.nan
            coverage_median = np.nan

        observation_uncertainty_corr = np.corrcoef(
            self.measurements.to_numpy().flatten(), self.measurement_uncertainties.to_numpy().flatten()
        )[0, -1]
        linearity_corr = np.corrcoef(np.arange(self.measurements.shape[0]), self.measurements.to_numpy().flatten())[
            0, -1
        ]

        rate_of_change = (
            LinearRegression()
            .fit(self.measurements.index.to_numpy().reshape(-1, 1), self.measurements.to_numpy().flatten())
            .coef_[0]
        )

        mean = self.measurements.mean()
        total = self.measurements.sum()

        data = np.zeros((21, 3))

        # Start
        data[0, 0] = start
        data[0, 1] = start - start_uncertainty
        data[0, 2] = start + start_uncertainty

        # End
        data[1, 0] = end
        data[1, 1] = end - end_uncertainty
        data[1, 2] = end + end_uncertainty

        # Difference
        data[2, 0] = difference
        data[2, 1] = difference - difference_uncertainty
        data[2, 2] = difference + difference_uncertainty

        # Maximum
        data[3, 0] = value_maximum
        data[3, 1] = value_maximum - value_maximum_uncertainty
        data[3, 2] = value_maximum + value_maximum_uncertainty

        # Minimum
        data[4, 0] = value_minimum
        data[4, 1] = value_minimum - value_minimum_uncertainty
        data[4, 2] = value_minimum + value_minimum_uncertainty

        # Percentiles (0.9, 0.75, 0.5, 0.25, 0.1)
        data[5, 0] = percentile_10
        data[5, 1] = np.nan
        data[5, 2] = np.nan

        data[6, 0] = percentile_25
        data[6, 1] = np.nan
        data[6, 2] = np.nan

        data[7, 0] = percentile_50
        data[7, 1] = np.nan
        data[7, 2] = np.nan

        data[8, 0] = percentile_75
        data[8, 1] = np.nan
        data[8, 2] = np.nan

        data[9, 0] = percentile_90
        data[9, 1] = np.nan
        data[9, 2] = np.nan

        # Coverage Maximum
        data[10, 0] = coverage_maximum
        data[10, 1] = np.nan
        data[10, 2] = np.nan

        # Coverage Minimum
        data[11, 0] = coverage_minimum
        data[11, 1] = np.nan
        data[11, 2] = np.nan

        # Coverage Range
        data[12, 0] = coverage_range
        data[12, 1] = np.nan
        data[12, 2] = np.nan

        # Coverage Mean
        data[13, 0] = coverage_mean
        data[13, 1] = np.nan
        data[13, 2] = np.nan

        # Coverage Median
        data[14, 0] = coverage_median
        data[14, 1] = np.nan
        data[14, 2] = np.nan

        # Coverage Std
        data[15, 0] = coverage_std
        data[15, 1] = np.nan
        data[15, 2] = np.nan

        # observation_uncertainty_corr
        data[16, 0] = observation_uncertainty_corr
        data[16, 1] = np.nan
        data[16, 2] = np.nan

        # linearity_corr
        data[17, 0] = linearity_corr
        data[17, 1] = np.nan
        data[17, 2] = np.nan

        # rate_of_change
        data[18, 0] = rate_of_change
        data[18, 1] = np.nan
        data[18, 2] = np.nan

        # mean
        data[19, 0] = mean
        data[19, 1] = np.nan
        data[19, 2] = np.nan

        # total
        data[20, 0] = total
        data[20, 1] = np.nan
        data[20, 2] = np.nan

        indexes = [
            "Initial Value",
            "Final Value",
            "Difference",
            "Maximum Value",
            "Minimum Value",
            "10th Percentile Value",
            "25th Percentile Value",
            "50th Percentile Value",
            "75th Percentile Value",
            "90th Percentile Value",
            "Coverage Maximum (%)",
            "Coverage Minimum (%)",
            "Coverage Range (%)",
            "Coverage Mean (%)",
            "Coverage Median (%)",
            "Coverage Standard Deviation (%)",
            "Observation/Uncertainty Correlation",
            "Observation Linearity (correlation coef.)",
            "rate of change",
            "Mean",
            "Total",
        ]

        summary_statistics = pd.DataFrame(
            data=data, index=indexes, columns=["Value", "Lower Uncertainty Bound", "Upper Uncertainty Bound"]
        )

        self.summary_stats = summary_statistics
