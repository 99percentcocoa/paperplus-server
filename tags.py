from tinydb import TinyDB, Query
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Sheets Database
# Columns:
#   1. Sheet ID (0, 1, 2, ...)
#   2. Sheet Name
#   3. Answer Key
#   4. Questions

db = TinyDB('worksheets.json')

BASE = 586
ORIENTATION_ID = 586

# this function arranges detection points so that they are in a clockwise order
import numpy as np

def sort_detections_clockwise(detections):
    # Extract centers as (x, y)
    centers = np.array([d.center for d in detections])
    ids = [d.tag_id for d in detections]

    # Compute centroid of all tag centers
    cx, cy = np.mean(centers, axis=0)

    # Compute angles of each point relative to centroid
    # atan2(y - cy, x - cx) gives angle from x-axis
    angles = np.arctan2(centers[:,1] - cy, centers[:,0] - cx)

    # Convert to degrees if you want to visualize
    # angles_deg = np.degrees(angles)

    # Sort by angle (clockwise)
    # Note: atan2 gives counterclockwise order by default,
    # so we sort descending for clockwise
    sorted_indices = np.argsort(angles)

    # Reorder detections and IDs
    detections_sorted = [detections[i] for i in sorted_indices]
    ids_sorted = [ids[i] for i in sorted_indices]
    logging.debug(f"Detected IDs: {ids_sorted}")

    return detections_sorted

def encode_worksheet_id(n: int):
    """Return tag IDs for TR, BR, BL given worksheet_id n."""
    if n >= BASE ** 3:
        raise ValueError(f"Max worksheet_id is {BASE**3 - 1}")
    ids = []
    for _ in range(3):
        ids.append(n % BASE)
        n //= BASE
    return ids  # [TR, BR, BL]


# Decode TR, BR, BL  →  worksheet_id
def decode_from_tags(tr: int, br: int, bl: int):
    """Return worksheet_id from three tag IDs."""
    return tr + br * BASE + bl * (BASE ** 2)


# Helper: rotate list clockwise
def rotate(lst, n):
    """Rotate list by n positions (clockwise)."""
    return lst[-n:] + lst[:-n]


# look up detected worksheet, and return correct order tag_ids
# returns worksheet ID and the correct order of tag_ids
def detect_orientation_and_decode(detection):
    """
    tag_ids: list of 4 detected tag IDs in clockwise order
             starting from any corner.
    Returns (worksheet_id, numRotations)
    """
    numRotations = 0

    for rot in range(4):
        # rot starts with 0
        rotated = rotate(detection, rot)
        tag_ids = [d.tag_id for d in rotated]
        numRotations += 1
        print(f"At rotation {numRotations}")
        if tag_ids[0] == ORIENTATION_ID:        # TL found

            worksheet_id = decode_from_tags(tag_ids[1], tag_ids[2], tag_ids[3])
            print(f"Scanned worksheet ID: {worksheet_id}")

            # check if worksheet id is in database
            if db.contains(doc_id=worksheet_id):
                print(f"Found worksheet id {worksheet_id}: {db.get(doc_id=worksheet_id).get('name', '')}")
                return worksheet_id, rotated
            else:
                print(f"Worksheet ID {worksheet_id} not found in database.")
                return None, None
    return None, None  # some error

if __name__ == "__main__":
    worksheet_id = 2
    print(encode_worksheet_id(2))