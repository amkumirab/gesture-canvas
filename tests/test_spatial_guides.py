import numpy as np

from gesture_canvas.spatial import SpatialGuide
from gesture_canvas.spatial_guides import depth_marker_y, draw_spatial_guides


def make_guide(z_position: float = 0) -> SpatialGuide:
    return SpatialGuide(
        origin=(120, 100),
        x_axis=(170, 100),
        y_axis=(120, 150),
        z_axis=(95, 80),
        z_position=z_position,
        minimum_z=-400,
        maximum_z=200,
    )


def test_depth_marker_places_near_above_far():
    assert depth_marker_y(make_guide(200), 100, 300) == 100
    assert depth_marker_y(make_guide(-400), 100, 300) == 300
    assert depth_marker_y(make_guide(-100), 100, 300) == 200


def test_depth_marker_validates_layout_and_range():
    with np.testing.assert_raises(ValueError):
        depth_marker_y(make_guide(), 100, 100)
    invalid = make_guide()
    invalid = SpatialGuide(
        origin=invalid.origin,
        x_axis=invalid.x_axis,
        y_axis=invalid.y_axis,
        z_axis=invalid.z_axis,
        z_position=0,
        minimum_z=1,
        maximum_z=1,
    )
    with np.testing.assert_raises(ValueError):
        depth_marker_y(invalid, 100, 300)


def test_spatial_guides_draw_axes_and_depth_ruler():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    draw_spatial_guides(frame, make_guide(), content_top=74)

    assert np.count_nonzero(frame) > 0
    assert np.count_nonzero(frame[:, -120:]) > 0


def test_spatial_guides_highlight_locked_rotation_axis():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_spatial_guides(
        frame,
        make_guide(),
        content_top=74,
        active_control="tilt_x",
    )

    x_axis_brightness = int(frame[100, 145].sum())
    y_axis_brightness = int(frame[125, 120].sum())
    assert x_axis_brightness > y_axis_brightness


def test_spatial_guides_skip_missing_or_tiny_views():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    draw_spatial_guides(frame, make_guide())
    draw_spatial_guides(frame, None)

    assert not np.any(frame)
