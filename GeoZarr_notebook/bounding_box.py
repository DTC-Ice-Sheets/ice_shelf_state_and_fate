"""A simple class for defining and retrieving extent related parameters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Union


@dataclass
class BoundingBox:
    """
    Defines any number of spatial constraints.

    Parameters
    ----------
    min_x : float, optional
        min x, by default None
    max_x : float, optional
        max x, by default None
    min_y : float, optional
        min y, by default None
    max_y : float, optional
        max y, by default None
    min_t : datetime or float, optional
        min t, by default None
    max_t : datetime or float, optional
        max t, by default None
    number_of_points : float, optional
        number of points in the bounding box, by default None
    min_z : float, optional
        min z, by default None
    max_z : float, optional
        max z, by default None

    Notes
    -----
    The x and y attributes can be used as longitude and latitude.
    """

    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    min_t: Optional[Union[float, datetime]] = None
    max_t: Optional[Union[float, datetime]] = None
    number_of_points: Optional[float] = None
    min_z: Optional[float] = None
    max_z: Optional[float] = None

    def __post_init__(self: BoundingBox) -> None:
        """Ensure max values are greater than min values while instantiating object."""
        # Input validation
        self.__class__._validate_min_max_pair(self.min_x, self.max_x)
        self.__class__._validate_min_max_pair(self.min_y, self.max_y)
        self.__class__._validate_min_max_pair(self.min_z, self.max_z)
        self.__class__._validate_min_max_pair(self.min_t, self.max_t)

    def __str__(self: BoundingBox) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        str
            A simple string summary of the defined extent.
        """
        output = ""
        if self.coordinate_extent_defined():
            output = f"min_x={self.min_x}, max_x={self.max_x}, min_y={self.min_y}, max_y={self.max_y}"

        if self.z_extent_defined():
            if output != "":
                output += ", "
            output += f"min_z={self.min_z}, max_z={self.max_z}"

        if self.date_extent_defined():
            if output != "":
                output += ", "
            output += f"min_t={self.min_t}, max_t={self.max_t}"

        if self.number_of_points is not None:
            if output != "":
                output += ", "
            output += f"number_of_points={self.number_of_points}"

        return output

    @staticmethod
    def _validate_min_max_pair(
        min_value: Union[datetime, float, None], max_value: Union[datetime, float, None]
    ) -> None:
        if (min_value is None) or (max_value is None):
            return

        if max_value < min_value:
            raise ValueError(f"Min value {min_value} cannot be greater than max {max_value}")

    def as_dict(self: BoundingBox) -> Dict[str, Union[None, float, datetime]]:
        """
        Return a dictionary of bounding box values.

        Returns
        -------
        Dict[str, Union[None, float, datetime]]
            Bounding box return
        """
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_t": self.min_t,
            "max_t": self.max_t,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "number_of_points": self.number_of_points,
        }

    @staticmethod
    def from_dict(extent_dict: Dict[str, Union[None, float, datetime]]) -> BoundingBox:
        """
        Build a BoundingBox object from a dictionary.

        Parameters
        ----------
        extent_dict : Dict[str, Union[None, float, datetime]]
            Dictionary of bounding box values

        Returns
        -------
        BoundingBox
            New BoundingBox instance
        """
        return BoundingBox(**extent_dict)

    def intersects(self: BoundingBox, bounding_box: BoundingBox) -> bool:
        """
        Check if instances intersects another BoundingBox instance. Touching edges does not count as an intersect.

        Parameters
        ----------
        bounding_box : BoundingBox
            Second test BoundingBox instance

        Returns
        -------
        bool
            True if instances overlap, otherwise false.
        """
        is_intersecting = True

        # For each pair of min/max values defined for bounding box, check if they possibly overlap target bounding box
        for self_min, self_max, bounding_box_min, bounding_box_max in [
            [self.min_x, self.max_x, bounding_box.min_x, bounding_box.max_x],
            [self.min_y, self.max_y, bounding_box.min_y, bounding_box.max_y],
            [self.min_z, self.max_z, bounding_box.min_z, bounding_box.max_z],
            [self.min_t, self.max_t, bounding_box.min_t, bounding_box.max_t],
        ]:
            if (self_min is not None) & (self_max is not None):
                is_intersecting = is_intersecting and not (
                    (self_max <= bounding_box_min) or (self_min >= bounding_box_max)
                )

        return is_intersecting

    def coordinate_extent_defined(self: BoundingBox) -> bool:
        """
        Check if a non-zero x/y coordinate extent exists.

        Returns
        -------
        bool
            True if extent defined
        """
        return (self.min_x is not None) & (self.max_y is not None) & (self.min_y is not None) & (self.max_y is not None)

    def date_extent_defined(self: BoundingBox) -> bool:
        """
        Check if a non-zero time extent exists.

        Returns
        -------
        bool
            True if extent defined
        """
        return (self.min_t is not None) & (self.max_t is not None)

    def z_extent_defined(self: BoundingBox) -> bool:
        """
        Check if a non-zero z (vertical) extent exists.

        Returns
        -------
        bool
            True if extent defined
        """
        return (self.min_z is not None) & (self.max_z is not None)
