"""A geospatially aware extension of the numpy array class."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform as tr
from affine import Affine
from numpy.typing import NDArray
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from use_case_3.raster_tools.bounding_box import BoundingBox
from use_case_3.raster_tools.helpers import make_grid_mesh_xy


class RasterNp(np.ndarray):
    """A geospatial aware numpy array.

    Usage is: arr_instance = RasterNp(np_array, bounding_box, resolution, [y_affine_negative=True], [proj4_code=''])
    """

    transform = None
    bounding_box = None
    resolution = None
    y_affine_negative = None
    proj4_code = None

    def _init(
        self: RasterNp,
        bounding_box: BoundingBox,
        resolution: float,
        y_affine_negative: bool,
        proj4_code: Union[str, None],
    ) -> None:
        self.bounding_box = self._get_corrected_bounding_box(bounding_box, resolution, y_affine_negative)
        self.resolution = resolution
        self.y_affine_negative = y_affine_negative
        self.proj4_code = proj4_code
        self.transform = RasterNp._get_transform(bounding_box, resolution, y_affine_negative)

    def __new__(
        cls: Any,
        input_array: NDArray[np.float64],
        bounding_box: BoundingBox,
        resolution: float,
        y_affine_negative: Optional[bool] = True,
        proj4_code: Optional[Union[str, None]] = None,
    ) -> RasterNp:
        """
        Create a RasterNp object.

        Parameters
        ----------
        input_array : NDArray[np.float64]
            Numpy data array of values, assumed to be regularly spaced in x/y coordinate space. Types other than float
            may work but aren't guaranteed to be supported.
        bounding_box : BoundingBox
            Spatial extent associated to the array. Note that providing this as center points of corner pixels is not
            ideal but it is handled. (c.f. _get_corrected_bounding_box)
        resolution : float
            The spacing between each pixel. Strictly redundant information for the creation of this object
            but it is usually known and helps catch obvious misunderstandings of raster extents.
        y_affine_negative : Optional[bool]
            If True the data is assumed to decrease in y coordinate as y index increases, by default True
        proj4_code : Optional[Union[str, None]]
            The CRS metadata for the extent, assumed to be in proj4 convention, by default None

        Returns
        -------
        RasterNp
            New class instance.
        """
        obj = np.asarray(input_array).view(cls)
        obj._init(bounding_box, resolution, y_affine_negative, proj4_code)
        return obj

    def _get_corrected_bounding_box(
        self: RasterNp, bounding_box: BoundingBox, resolution: float, _y_affine_negative: bool
    ) -> BoundingBox:
        # Verify / correct dimensions
        expected_x_count = round((bounding_box.max_x - bounding_box.min_x) / resolution)
        expected_y_count = round((bounding_box.max_y - bounding_box.min_y) / resolution)

        if self.shape[1] - expected_x_count == 1 and self.shape[0] - expected_y_count == 1:
            bounding_box.min_x = bounding_box.min_x - resolution / 2
            bounding_box.max_x = bounding_box.max_x + resolution / 2
            bounding_box.min_y = bounding_box.min_y - resolution / 2
            bounding_box.max_y = bounding_box.max_y + resolution / 2

        expected_x_count = round((bounding_box.max_x - bounding_box.min_x) / resolution)
        expected_y_count = round((bounding_box.max_y - bounding_box.min_y) / resolution)

        if expected_x_count != self.shape[1] or expected_y_count != self.shape[0]:
            raise ValueError("Extent and resolution do not match array dimensions")
        return bounding_box

    def _get_resolution_rounded_bounding_box(self: RasterNp, bounding_box: BoundingBox) -> BoundingBox:
        reference_x, reference_y = self.bounding_box.min_x, self.bounding_box.min_y
        x_min = reference_x - math.ceil((reference_x - bounding_box.min_x) / self.resolution) * self.resolution
        y_min = reference_y - math.ceil((reference_y - bounding_box.min_y) / self.resolution) * self.resolution
        x_max = reference_x + math.ceil((bounding_box.max_x - reference_x) / self.resolution) * self.resolution
        y_max = reference_y + math.ceil((bounding_box.max_y - reference_y) / self.resolution) * self.resolution
        return BoundingBox(x_min, x_max, y_min, y_max)

    @staticmethod
    def _get_transform(bounding_box: BoundingBox, resolution: float, y_affine_negative: bool) -> Affine:
        if y_affine_negative:
            transform = Affine.translation(bounding_box.min_x, bounding_box.max_y) * Affine.scale(
                resolution, -resolution
            )
        else:
            transform = Affine.translation(bounding_box.min_x, bounding_box.min_y) * Affine.scale(
                resolution, resolution
            )
        return transform

    def yx_from_rows_cols(
        self: RasterNp, y_idx: Union[int, NDArray], x_idx: Union[int, NDArray]
    ) -> Tuple[Union[float, NDArray], Union[float, NDArray]]:
        """
        Get y, x coordinates (centre points) for corresponding index values in the array data.

        return shape is the same as the input shape (can be single values or nd-arrays)

        Parameters
        ----------
        y_idx : Union[int, NDArray]
            y index or indices to retrieve coordinates for.
        x_idx : Union[int, NDArray]
            x index or indices to retrieve coordinates for.

        Returns
        -------
        Tuple[Union[float, NDArray], Union[float, NDArray]]
            y coordinate value(s)
            x coordinate value(s)
        """
        x, y = tr.xy(self.transform, y_idx, x_idx)
        return y, x

    def rows_cols_from_yx(
        self: RasterNp, y: Union[float, NDArray, Tuple, List], x: Union[float, NDArray, Tuple, List]
    ) -> Tuple[Union[int, np.int64, NDArray], Union[int, np.int64, NDArray]]:
        """
        Return row, col array indices for for real y, x coordinates values.

        return shape is the same as the input shape (can be single values or nd-arrays)

        The rasterio function for this is too slow unfortunately, otherwise the following is a neater solution.
        e.g y_idx, x_idx = tr.rowcol(self.transform, x, y)

        Parameters
        ----------
        y : Union[float, NDArray, Tuple, List]
            y coordinates to retrieve index/indices for.
        x : Union[float, NDArray, Tuple, List]
            x coordinates to retrieve index/indices for.

        Returns
        -------
        Tuple[Union[int, np.int64, NDArray], Union[int, np.int64, NDArray]]
            y index or indices.
            x index or indices.
        """
        if x is not np.iterable:
            x_iter = [x]
        else:
            x_iter = x
        if y is not np.iterable:
            y_iter = [y]
        else:
            y_iter = y

        x_idx, y_idx = np.floor(~self.transform * np.array(*zip(x_iter, y_iter))).astype(int)

        return y_idx, x_idx

    def yx_gridpoints(self: RasterNp) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return arrays of y and x coordinates for the entire raster.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            List of centre point coordinates for y axis
            List of centre point coordinates for x axis
        """
        y = np.arange(self.bounding_box.min_y + self.resolution / 2.0, self.bounding_box.max_y, self.resolution)
        x = np.arange(self.bounding_box.min_x + self.resolution / 2.0, self.bounding_box.max_x, self.resolution)

        return y, x

    def new_extent_copy(
        self: RasterNp,
        bounding_box: BoundingBox,
        fill_to_extent: Optional[bool] = False,
        fill_value: Optional[float] = np.nan,
    ) -> RasterNp:
        """
        Return a new RasterNp object, resized to the new extent.

        Parameters
        ----------
        bounding_box : BoundingBox
            Target extent for new array.
        fill_to_extent : Optional[bool]
            If True and extent is greater than source image, fill with fill_value, by default False
        fill_value : Optional[float]
            Default value for pixels outside the extent of the existing array (e.g. if resizing to a larger extent),
            by default np.nan

        Returns
        -------
        RasterNp
            New resized RasterNp object
        """
        bounding_box = self._get_resolution_rounded_bounding_box(bounding_box)
        overlap_extent = BoundingBox(
            min_x=max(bounding_box.min_x, self.bounding_box.min_x),
            max_x=min(bounding_box.max_x, self.bounding_box.max_x),
            min_y=max(bounding_box.min_y, self.bounding_box.min_y),
            max_y=min(bounding_box.max_y, self.bounding_box.max_y),
        )

        if not fill_to_extent:
            bounding_box = overlap_extent

        xs = np.arange(bounding_box.min_x, bounding_box.max_x - self.resolution * 0.1, self.resolution)
        ys = np.arange(bounding_box.min_y, bounding_box.max_y - self.resolution * 0.1, self.resolution)
        new_raster = RasterNp(
            np.ones((len(ys), len(xs))) * fill_value,
            bounding_box,
            self.resolution,
            self.y_affine_negative,
            self.proj4_code,
        )

        xs_extract = np.arange(
            overlap_extent.min_x + self.resolution * 0.5, overlap_extent.max_x + self.resolution * 0.4, self.resolution
        )
        ys_extract = np.arange(
            overlap_extent.min_y + self.resolution * 0.5, overlap_extent.max_y + self.resolution * 0.4, self.resolution
        )

        if self.y_affine_negative:
            ys = ys[::-1]
            ys_extract = ys_extract[::-1]

        y_idx_extract, _ = self.rows_cols_from_yx(ys_extract, np.repeat(xs_extract[0], len(ys_extract)))
        _, x_idx_extract = self.rows_cols_from_yx(np.repeat(ys_extract[0], len(xs_extract)), xs_extract)
        y_idx_assign, _ = new_raster.rows_cols_from_yx(ys_extract, np.repeat(xs_extract[0], len(ys_extract)))
        _, x_idx_assign = new_raster.rows_cols_from_yx(np.repeat(ys_extract[0], len(xs_extract)), xs_extract)

        new_raster[y_idx_assign[0] : y_idx_assign[-1] + 1, x_idx_assign[0] : x_idx_assign[-1] + 1] = self[
            y_idx_extract[0] : y_idx_extract[-1] + 1, x_idx_extract[0] : x_idx_extract[-1] + 1
        ]

        return new_raster

    def at_yx(self: RasterNp, y: Union[NDArray[np.float64], List], x: Union[NDArray[np.float64], List]) -> NDArray:
        """
        Return values from raster for given y and x coordinate lists.

        Note, x and y expected to be the same length and no interpolation is applied.

        Parameters
        ----------
        y : Union[NDArray[np.float64], List]
            List of y coordinates to extract values for.
        x : Union[NDArray[np.float64], List]
            List of x coordinates to extract values for.

        Returns
        -------
        NDArray
            2D Array of values for the specified coordinates.
        """
        y_idx, x_idx = self.rows_cols_from_yx(y, x)
        return self[y_idx, x_idx]

    def get_centre_points(self: RasterNp) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get x coords, y coords, and values in an x-oriented array for mesh grid consistency.

        This was originally for backwards compatibility for use in the EOLIS processing.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            x centre point coordinates.
            y centre point coordinates.
            Array values in x-oriented order.
        """
        y, x = self.yx_gridpoints()
        if self.y_affine_negative:
            values = self[::-1]
        else:
            values = self

        return x, y, np.transpose(values)

    def save_as_tiff(
        self: RasterNp,
        file_path: str,
        data_type: Optional[str] = "float32",
        no_data: Optional[float] = None,
        **kwargs: Optional[Union[dict[str, Any], None, str]],
    ) -> None:
        """
        Save object as a geotiff.

        Parameters
        ----------
        file_path : str
            Output file path.
        data_type : Optional[str]
            Override the output number type, by default 'float32'
        no_data : Optional[float]
            Override the no-data value, by default None
        kwargs : Optional[Union[dict[str, Any], None, str]]
            Keyword arguments for rasterio.open. These are passed to format drivers as directives
            for creating or interpreting datasets.
        """
        # set default compression for backwards compatibility if not explicitly specified by user
        if "compress" not in list(kwargs.keys()):
            kwargs["compress"] = "lzw"

        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=self.shape[0],
            width=self.shape[1],
            count=1,
            dtype=data_type,
            crs=self.proj4_code,
            transform=self.transform,
            nodata=no_data,
            **kwargs,
        ) as rst:
            rst.write_band(1, self.astype(data_type))

    @staticmethod
    def _round_window_include_all_partial_pixels(window: Window, full_raster_window: Window) -> Window:
        # first round the window to give the maximum area
        top = math.floor(window.row_off)
        left = math.floor(window.col_off)
        bottom = math.ceil(window.row_off + window.height)
        right = math.ceil(window.col_off + window.width)

        # then ensure the window is still within the window of the entire raster
        top = full_raster_window.row_off if top < full_raster_window.row_off else top
        left = full_raster_window.col_off if left < full_raster_window.col_off else left
        bottom = (
            full_raster_window.row_off + full_raster_window.height
            if bottom > full_raster_window.row_off + full_raster_window.height
            else bottom
        )
        right = (
            full_raster_window.col_off + full_raster_window.width
            if right > full_raster_window.col_off + full_raster_window.width
            else right
        )

        return Window(left, top, right - left, bottom - top)

    @staticmethod
    def _bounding_box_intersects_bounds(bounds: Tuple[float, float, float, float], bounding_box: BoundingBox) -> bool:
        """
        Check if any data exists in the bounds for the given extent.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            Rasterio bounds
        bounding_box : BoundingBox
            Bounding box extent.

        Returns
        -------
        bool
            True if extents overlap.
        """
        bounds_as_extent = BoundingBox(
            bounds.left, bounds.right, min(bounds.bottom, bounds.top), max(bounds.bottom, bounds.top)
        )
        return bounding_box.intersects(bounds_as_extent)

    @staticmethod
    def load_from_tiff(file_path: str, band: Optional[int] = 1, bounding_box: Optional[BoundingBox] = None) -> RasterNp:
        """
        Build a RasterNp object from a tiff file.

        This may work with other geo-image types, too, but is untested.

        Parameters
        ----------
        file_path : str
            The file path to attempt to load.
        band : Optional[int]
            Which raster band to load, by default 1 (the first one)
        bounding_box : Optional[BoundingBox]
            If a bounding box is passed, only a sub-area of the tiff is loaded. If the bounding box bounds are greater
            than the image bounds, they are trimmed down to the image bounds. All partial pixels touched by the bounding
            box are returned. If None, the entire tiff is loaded. By default None.

        Returns
        -------
        RasterNp
            New RasterNp instance.

        Raises
        ------
        ValueError
            If requested extent does not overlap the file extent.
        """
        with rasterio.open(file_path) as rst:
            if bounding_box is None:
                data = rst.read(band)
                transform = rst.transform
                bounds = rst.bounds[0], rst.bounds[1], rst.bounds[2], rst.bounds[3]
                no_data_mask = rst.read_masks(band)
            else:
                if not RasterNp._bounding_box_intersects_bounds(rst.bounds, bounding_box):
                    raise ValueError(
                        "Bounding box does not overlap image. Bounding box: {}, File: {}".format(
                            bounding_box, file_path
                        )
                    )
                full_raster_window = from_bounds(
                    rst.bounds.left, rst.bounds.bottom, rst.bounds.right, rst.bounds.top, rst.transform
                )
                window = RasterNp._from_bounds(rst.bounds, bounding_box, rst.transform)
                window = RasterNp._round_window_include_all_partial_pixels(window, full_raster_window)
                data = rst.read(band, window=window)
                no_data_mask = rst.read_masks(band, window=window)
                transform = rasterio.windows.transform(window, rst.transform)
                bounds = rasterio.windows.bounds(window, rst.transform)

            # For float types, set no value = nan
            if rst.nodata is not None:
                if "int" in str(data.dtype):
                    data = data.astype("float32")
                if "float" in str(data.dtype):
                    data[no_data_mask == 0] = np.nan

            bounding_box = BoundingBox(
                min_x=bounds[0],
                max_x=bounds[2],
                min_y=np.min((bounds[1], bounds[3])),
                max_y=np.max((bounds[1], bounds[3])),
            )
            resolution = transform.a
            y_affine_negative = transform.e < 0
            if rst.crs is None:
                proj4_code = None
            else:
                proj4_code = rst.crs.to_proj4()

        return RasterNp(data, bounding_box, resolution, y_affine_negative, proj4_code)

    def convert_to_dataframe(self: RasterNp, value_name: Optional[str] = "value") -> pd.DataFrame:
        """
        Convert object to a dataframe with x coordinates, y coordinates, and values.

        Parameters
        ----------
        value_name : Optional[str]
            The new column name for the data in the DataFrame, by default 'value'

        Returns
        -------
        pd.DataFrame
            New DataFrame instance with 'x', 'y', value_name columns.
        """
        x, y, data = self.get_centre_points()
        x, y = make_grid_mesh_xy(x, y)
        df = pd.DataFrame(data={"x": x, "y": y, value_name: data.flatten()})
        return df

    @staticmethod
    def _from_bounds(bounds: Tuple[float, float, float, float], new_extent: BoundingBox, transform: Affine) -> Window:
        top = bounds.top if new_extent.max_y > bounds.top else new_extent.max_y
        left = bounds.left if new_extent.min_x < bounds.left else new_extent.min_x
        bottom = bounds.bottom if new_extent.min_y < bounds.bottom else new_extent.min_y
        right = bounds.right if new_extent.max_x > bounds.right else new_extent.max_x

        return from_bounds(left, bottom, right, top, transform)

    @staticmethod
    def bounding_box_intersects_tiff(file_path: str, bounding_box: BoundingBox) -> bool:
        """
        Check if any data exists in the tiff for the given extent.

        Parameters
        ----------
        file_path : str
            Tiff file path.
        bounding_box : BoundingBox
            Extent to check tiff data exists for.

        Returns
        -------
        bool
            True if extent overlaps tiff.
        """
        with rasterio.open(file_path) as rst:
            bounds = rst.bounds

        return RasterNp._bounding_box_intersects_bounds(bounds, bounding_box)

    def mask_from_file(
        self: RasterNp,
        file_path: str,
        include_mask_value: Optional[float] = 1,
        outside_mask_return_value: Optional[float] = np.nan,
    ) -> None:
        """
        Mask object in place from another geotiff file.

        Parameters
        ----------
        file_path : str
            File path of mask file.
        include_mask_value : Optional[float]
            The in-mask value in the mask geotiff, by default 1.
        outside_mask_return_value : Optional[float]
            The value assinged to data found outside of the mask, by default np.nan. note - the default
            outside_mask_return_value value of np.nan does not work with int-based arrays.
        """
        # For each point in current file, get mask value and mask nan accordingly
        y, x = self.yx_gridpoints()
        x, y = make_grid_mesh_xy(x, y)

        # Load intersecting mask tiff
        mask = RasterNp.load_from_tiff(file_path, bounding_box=self.bounding_box)
        # Now force mask extent to cover the same area as self.extent
        mask = mask.new_extent_copy(self.bounding_box, fill_to_extent=True, fill_value=outside_mask_return_value)
        mask_values = mask.at_yx(y, x)
        mask_values = mask_values.reshape(-1, self.shape[0]).swapaxes(0, 1)
        mask_values = mask_values == include_mask_value

        # Apply mask
        if self.y_affine_negative:
            mask_values = mask_values[::-1]

        self[~mask_values] = outside_mask_return_value

    def mask_from_raster_np(
        self: RasterNp,
        mask: RasterNp,
        include_mask_value: Optional[Union[float, bool]] = 1,
        outside_mask_return_value: Optional[float] = np.nan,
    ) -> None:
        """
        Mask object in place from a np.ndarray mask.

        Parameters
        ----------
        mask : RasterNp
            The mask represented as a RasterNp.
        include_mask_value : Optional[float]
            The in-mask value in the mask geotiff, by default 1.
        outside_mask_return_value : Optional[float]
            The value assigned to data found outside of the mask, by default np.nan. note - the default
            outside_mask_return_value value of np.nan does not work with int-based arrays.

        TODO: Check and return error when using int-based arrays.
        """
        # For each point in current file, get mask value and mask nan accordingly
        y, x = self.yx_gridpoints()
        x, y = make_grid_mesh_xy(x, y)

        # Now force mask extent to cover the same area as self.extent
        mask = mask.new_extent_copy(self.bounding_box, fill_to_extent=True, fill_value=outside_mask_return_value)
        mask_values = mask.at_yx(y, x)
        mask_values = mask_values.reshape(-1, self.shape[0]).swapaxes(0, 1)
        mask_values = mask_values == include_mask_value

        # Apply mask
        if self.y_affine_negative:
            mask_values = mask_values[::-1]

        self[~mask_values] = outside_mask_return_value

    @staticmethod
    def rasterise_mask_from_geometry(
        geometries: Union[list, gpd.GeoDataFrame],
        y_min: Union[int, float],
        y_max: Union[int, float],
        x_min: Union[int, float],
        x_max: Union[int, float],
        resolution: Union[int, float],
        y_affine_negative: bool = False,
    ) -> RasterNp:
        """
        Rasterise a mask based on a geometry.

        Parameters
        ----------
        geometries : Union[list, gpd.G
            A list of multipolygons
        y_min : Union[int, float]
            The extent y_min.
        y_max : Union[int, float]
            The extent y_max.
        x_min : Union[int, float]
            The extent x_min.
        x_max : Union[int, float]
            The extent x_max.
        resolution : Union[int, float]
            The resolution.
        y_affine_negative : bool, optional
            Whether the raster is y affine negative, by default False

        Returns
        -------
        RasterNp
            The raster mask.
        """
        y = int(np.ceil((y_max - y_min) / resolution))
        x = int(np.ceil((x_max - x_min) / resolution))

        bounding_box = BoundingBox(min_y=y_min, max_y=y_max, min_x=x_min, max_x=x_max)

        transform = RasterNp._get_transform(
            bounding_box=bounding_box, resolution=resolution, y_affine_negative=y_affine_negative
        )

        mask = geometry_mask(
            geometries=geometries["geometry"], out_shape=(y, x), transform=transform, invert=True, all_touched=True
        ).astype(int)
        mask = RasterNp(
            input_array=mask, bounding_box=bounding_box, resolution=resolution, y_affine_negative=y_affine_negative
        )

        return mask
