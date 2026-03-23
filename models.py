from dataclasses import dataclass, field
import random
from typing import List, Optional
from pathlib import Path

from config import SETTINGS

import cv2
import numpy as np
from pupil_apriltags import Detection as AprilTagDetection
from PIL.Image import Image as PILImage

@dataclass
class InputImageMeta:
    """Metadata for an input image."""
    image_path: Optional[Path | str] = None
    image_array: Optional[np.ndarray] = None
    image_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate that either image_path or image_array is provided."""
        if self.image_path is None and self.image_array is None:
            raise ValueError("Either image_path or image_array must be provided")
        if self.image_path is not None and self.image_array is not None:
            raise ValueError("Only one of image_path or image_array should be provided")
        
        # If image_path is provided and image_array is None, load the image
        if self.image_path is not None and self.image_array is None:
            self.load_bytes(str(self.image_path))
    
    def load_bytes(self, image_path: str) -> bool:
        """Load the image from filepath using cv2.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image loaded successfully, False otherwise
        """
        self.image_path = image_path
        self.image_array = cv2.imread(image_path)
        return self.image_array is not None
    
    def copy(self) -> 'InputImageMeta':
        """Create a copy of this InputImageMeta with only the image array.
        
        The copy will not include the original filepath, only the image data.
        This is useful for creating derivative images (e.g., processed versions)
        that shouldn't be associated with the original file path.
        
        Returns:
            A new InputImageMeta object with copied image array and no path.
        """
        if self.image_array is None:
            raise ValueError("Cannot copy InputImageMeta: image_array is None")
        
        return InputImageMeta(image_array=self.image_array.copy())
    
    def save(self, output_path: Optional[str] = None) -> bool:
        """Save the image array to disk.
        
        Args:
            output_path: Optional path to save the image to. If None, uses image_path.
            
        Returns:
            True if image saved successfully, False otherwise.
            
        Raises:
            ValueError: If image_array is None or if both output_path and image_path are None.
            
        Example:
            >>> img = InputImageMeta(image_path="input.jpg")
            >>> img.image_array = cv2.GaussianBlur(img.image_array, (5, 5), 0)
            >>> img.save("output.jpg")  # Save to different path
            >>> img.save()  # Save back to original path
        """
        if self.image_array is None:
            raise ValueError("Cannot save: image_array is None")
        
        path_to_use = output_path if output_path is not None else self.image_path
        
        if path_to_use is None:
            raise ValueError("Cannot save: no output_path provided and image_path is None")
        
        return cv2.imwrite(str(path_to_use), self.image_array)

@dataclass
class DetectionResult:
    """Results from AprilTag detection on an image."""
    input_image: InputImageMeta
    detections: List[AprilTagDetection]
    tag_family: str = None
    tag_ids: List[int] = field(init=False)
    sorted_detections: List[AprilTagDetection] = field(init=False)
    
    def __post_init__(self):
        """Extract tag IDs from detections and sort detections clockwise."""
        # Validate detections based on tag family
        num_detections = len(self.detections)
        
        if self.tag_family == "36h11":
            if num_detections < 4:
                raise ValueError(f"Tag family '36h11' requires at least 4 detections, but got {num_detections}")
            
            # Lazy import to avoid circular dependency
            from services.image_service import sort_detections_clockwise
            self.sorted_detections = sort_detections_clockwise(self.detections)

        elif self.tag_family == "25h9":
            required_detections = SETTINGS.NUM_ROW_TAGS
            if num_detections < required_detections:
                raise ValueError(f"Tag family '25h9' requires at least {required_detections} detections, but got {num_detections}")
            
            # Filter out detections with tag_id outside valid range [0, required_detections-1]
            valid_detections = [d for d in self.detections if 0 <= d.tag_id <= required_detections]
            self.detections = valid_detections
            
            # Sort 25h9 detections from top to bottom by Y-coordinate
            self.sorted_detections = sorted(self.detections, key=lambda d: d.center[1])
        
        # save tag_ids as list of int
        self.tag_ids = [detection.tag_id for detection in self.detections]
           
    
    # def sort_detections_clockwise(self) -> List[AprilTagDetection]:
    #     """Return detections ordered as: top-left, top-right, bottom-right, bottom-left.
        
    #     If there are exactly 4 detections (e.g., corner tags), we sort them
    #     by splitting into top and bottom pairs using the Y coordinate, then
    #     ordering each pair by X. For other counts, we fall back to angle-based
    #     ordering around the centroid.
        
    #     Returns:
    #         List of AprilTagDetection ordered for perspective transforms.
    #     """
    #     if len(self.detections) < 4:
    #         raise ValueError("At least 4 detections are required to sort.")

    #     centers = np.array([d.center for d in self.detections], dtype=float)

    #     # Corner ordering only when we have exactly 4 detections
    #     if len(self.detections) == 4:
    #         ys = centers[:, 1]
    #         xs = centers[:, 0]

    #         # Indices sorted by Y (top to bottom)
    #         y_sorted = np.argsort(ys)
    #         top_idx = y_sorted[:2]
    #         bottom_idx = y_sorted[2:]

    #         # Sort top pair by X ascending -> TL, TR
    #         top_order = top_idx[np.argsort(xs[top_idx])]

    #         # Sort bottom pair by X ascending -> BL, BR; we want BR, BL
    #         bottom_by_x = bottom_idx[np.argsort(xs[bottom_idx])]
    #         bl_idx, br_idx = bottom_by_x[0], bottom_by_x[1]

    #         ordered_indices = [top_order[0], top_order[1], br_idx, bl_idx]
    #         return [self.detections[i] for i in ordered_indices]

    #     # Fallback: angle-based order around centroid for non-4 counts
    #     cx, cy = np.mean(centers, axis=0)
    #     angles = np.arctan2(centers[:, 1] - cy, centers[:, 0] - cx)
    #     sorted_indices = np.argsort(angles)
    #     return [self.detections[i] for i in sorted_indices]


@dataclass
class WorksheetTemplate:
    """Metadata for a worksheet template."""
    input_image: InputImageMeta
    cropped_image: Optional[InputImageMeta] = None
    preprocessed_image: Optional[InputImageMeta] = None
    corner_detections: Optional[DetectionResult] = None
    row_detections: Optional[DetectionResult] = None
    debug_image: Optional[InputImageMeta] = None
    checked_image: Optional[PILImage] = None
    checked_image_url: Optional[str] = None
    num_questions: Optional[int] = None
    worksheet_id: Optional[int] = None
    marked_answers: Optional[List[str]] = None
    answer_key: Optional[List[str]] = None
    score: Optional[List[int]] = None

@dataclass
class ROI:
    """Data class to represent a Region of Interest (ROI) for a question."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def width(self) -> int:
        """Calculate the width of the ROI."""
        return self.x2 - self.x1
    
    def height(self) -> int:
        """Calculate the height of the ROI."""
        return self.y2 - self.y1

@dataclass
class ContourData:
    """Data class to hold contour information."""
    contour: np.ndarray
    area: float = field(init=False)
    perimeter: float = field(init=False)
    circularity: float = field(init=False)
    
    def __post_init__(self):
        """Calculate area, perimeter, and circularity from contour."""
        self.area = cv2.contourArea(self.contour)
        self.perimeter = cv2.arcLength(self.contour, closed=True)
        
        # Calculate circularity (4π * area / perimeter²)
        if self.perimeter > 0:
            self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2)
        else:
            self.circularity = 0.0
    
    def get_global_contour(self, offset_x: int, offset_y: int) -> np.ndarray:
        """Get the contour coordinates in the global image space by applying an offset."""
        return self.contour + np.array([[[offset_x, offset_y]]])

@dataclass
class Question:
    """Data class to track question information."""
    index: int
    question_text: str
    skill_code: str
    options: list
    answer: int # 1-based index of the correct answer in options, to be converted later
    possible_distractors: list
    correct_option: str = None # will be filled in choose_distractors

    def choose_distractors(self) -> list:
        """
        Choose 3 random distractors from all possible distractors
        and assign them random positions along with the correct answer.
        
        Returns:
            list: A list of 4 options with the correct answer at a random position.
        """
        # Get the correct answer from the options at the answer index
        correct_answer = self.options[self.answer - 1]

        # Keep only non-negative distractors and avoid duplicates of the correct answer.
        filtered_distractors = [
            d for d in self.possible_distractors
            if _is_non_negative_option(d) and d != correct_answer
        ]

        if len(filtered_distractors) < 3:
            raise ValueError("Need at least 3 non-negative possible distractors")

        # Choose 3 random distractors
        chosen_distractors = random.sample(filtered_distractors, 3)
        
        # Create a list with the correct answer and chosen distractors
        all_options = [correct_answer] + chosen_distractors
        
        # Shuffle the options
        random.shuffle(all_options)
        
        # Find the new position (1-4) of the correct answer
        new_answer_position = all_options.index(correct_answer) + 1
        
        # Update the object
        self.options = all_options
        self.answer = new_answer_position
        
        return all_options

# helper for Question.choose_distractors to filter out invalid distractors    
def _is_non_negative_option(value):
    """Return True only for non-negative answer/option values."""
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        if "R" in s:
            q, sep, r = s.partition("R")
            if not sep:
                return False
            try:
                return int(q) >= 0 and int(r) >= 0
            except ValueError:
                return False
        try:
            return int(s) >= 0
        except ValueError:
            return False
    return False

# helper for generating random digits for dummy distractors in case of insufficient valid distractors
def _rand_digit(exclude_zero=False):
    if exclude_zero:
        return random.randint(1, 9)
    return random.randint(0, 9)