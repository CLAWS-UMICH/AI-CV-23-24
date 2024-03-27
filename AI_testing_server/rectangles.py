import itertools
import math
import numpy as np

def angle_between_points(p1, p2, p3):
    """Calculate the angle between three points."""
    # Calculate vectors between points
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    # Calculate dot product and lengths
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    length_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    length_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # Calculate angle (in radians)
    angle_rad = math.acos(dot_product / (length_v1 * length_v2))

    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def slope(p1, p2):
    """Calculate the slope of the line passing through two points."""
    if p1[0] == p2[0]:
        return float('inf')  # Vertical line, slope is infinite
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def is_parallel(slope1, slope2):
    """Check if two slopes are parallel within a tolerance."""
    return abs(slope1 - slope2)

def COM_test(points):
    cx = sum(point[0] for point in points) / 4
    cy = sum(point[1] for point in points) / 4

    dd = [(cx - point[0]) ** 2 + (cy - point[1]) ** 2 for point in points]

    dex = np.array(dd)

    return np.linalg.norm(dex-np.average(dex))

def find_rectangle(points):
    """Find the four points most likely to form a rectangle."""
    min_rectangle_score = 10**10
    best_rectangle_points = None

    # Iterate through all combinations of 4 points
    for rectangle_points in itertools.permutations(points, 4):        
        # Calculate the total score for this rectangle        
        # rectangle_score = COM_test(rectangle_points)
        rectangle_score = 0

        # ANGLE SCORE
        for i in range(4):
            p1, p2, p3 = rectangle_points[i], rectangle_points[(i + 1) % 4], rectangle_points[(i + 2) % 4]
            angle = angle_between_points(p1, p2, p3)            
            # Add the angle to the total score
            rectangle_score += abs(90 - angle)

        # PARALLEL SCORE
        p1, p2, p3, p4 = rectangle_points
         # Calculate the slopes of the opposite sides
        slope1 = slope(p1, p2)
        slope2 = slope(p3, p4)

        # Check how parallel the lines are
        rectangle_score += is_parallel(slope1, slope2)

        # Update the best rectangle if this one has a higher score
        if rectangle_score < min_rectangle_score:
            min_rectangle_score = rectangle_score
            best_rectangle_points = rectangle_points

    if best_rectangle_points:
        best_rectangle_points_sortedx = sorted(best_rectangle_points, key=lambda coord: coord[0])
        best_rectangle_points_sortedy = sorted(best_rectangle_points_sortedx[:2], key=lambda coord: coord[1]) + sorted(best_rectangle_points_sortedx[2:], key=lambda coord: coord[1])
    else:
        best_rectangle_points_sortedy = None

    return best_rectangle_points_sortedy