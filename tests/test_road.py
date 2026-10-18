import pytest

from road import POINT_SPACING, build_center_line, nearest_point_index

START = (445, 600)
UP = 90 # The pen starts pointing up (0° = right, angles grow counterclockwise)


def test_straight_piece_goes_up():
    points = build_center_line([(1000, 0)], *START, UP)
    assert len(points) == 1000 // POINT_SPACING + 1 # One point every POINT_SPACING pixels, plus the starting point
    assert points[0] == START
    assert points[-1] == pytest.approx((445, -400)) # 1000 pixels higher, same x


def test_points_are_evenly_spaced():
    points = build_center_line([(300, 0), (300, -90), (300, 45)], *START, UP)
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        assert ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 == pytest.approx(POINT_SPACING)


def test_negative_degrees_turn_right_and_positive_turn_left():
    right = build_center_line([(300, -90)], *START, UP)
    left = build_center_line([(300, 90)], *START, UP)
    assert right[-1][0] > START[0] # Ended up to the right
    assert left[-1][0] < START[0] # Ended up to the left
    assert right[-1][1] == pytest.approx(left[-1][1]) # Mirror images: same height


def test_pieces_continue_where_the_previous_one_ended():
    points = build_center_line([(500, 0), (500, 0)], *START, UP)
    assert points[-1] == pytest.approx((445, -400)) # Two straight pieces = one long straight road


def test_nearest_point_follows_the_distance_along_the_road():
    points = build_center_line([(1000, 0)], *START, UP)
    assert nearest_point_index(points, 445, 600) == 0 # At the start
    assert nearest_point_index(points, 445, 100) == 100 # 500 px ahead = point 100
    assert nearest_point_index(points, 500, 100) == 100 # Moving sideways doesn't change it
    assert nearest_point_index(points, 445, -1000) == len(points) - 1 # Past the end: the last point
