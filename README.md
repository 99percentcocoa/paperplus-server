# Offline Pipeline Documentation

## Overview

The **Offline Pipeline** is a command-line tool for processing worksheet images locally without the WhatsApp messaging infrastructure. It performs the same OMR (Optical Mark Recognition) detection and grading as the main message service, saving dewarped, debug, and checked images to organized output folders.

This tool is useful for:
- Batch processing multiple worksheets
- Testing and debugging worksheet processing
- Local analysis without server connectivity
- Generating debug visualizations for troubleshooting

## Features

✅ **Complete OMR Pipeline**
- AprilTag corner detection and image dewarping
- Document cleaning and preprocessing
- Question mark detection and grading
- Bubble fill ratio analysis with circularity checking

✅ **Organized Output**
- Automatically creates output subdirectories
- Saves preprocessed (dewarped) images
- Saves debug images with detection overlays
- Saves graded images with answer marks and scores
- Generates results.json summary

✅ **Flexible Processing**
- Process single images or entire folders
- Supports multiple image formats (JPG, PNG, BMP, TIFF)
- Detailed logging with optional verbose mode
- Graceful error handling with informative messages

## Installation

### Prerequisites

Ensure you have Python 3.10+ and all required dependencies installed:

```bash
pip install -r requirements.txt
```

### Required Dependencies

The pipeline requires:
- `opencv-python` - Image processing
- `pillow` - PIL image library
- `pupil-apriltags` - AprilTag detection
- `tinydb` - Worksheet template database
- `numpy` - Numerical operations

All dependencies are listed in `requirements.txt`.

## Usage

### Basic Usage

Process a single image:

```bash
python offline_pipeline.py --input worksheet.jpg --output ./results/
```

Process all images in a folder:

```bash
python offline_pipeline.py --input images/ --output ./results/
```

### Command-Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | ✓ | Path to input image file or folder containing images |
| `--output` | `-o` | ✓ | Output folder where results will be saved |
| `--verbose` | `-v` | ✗ | Enable detailed logging output (debug mode) |

### Examples

**Single image processing:**
```bash
python offline_pipeline.py --input ./test_image.jpg --output ./results/
```

**Batch processing with verbose output:**
```bash
python offline_pipeline.py --input ./worksheets/ --output ./batch_results/ --verbose
```

**Processing from a different directory:**
```bash
cd /path/to/project && python offline_pipeline.py -i ./images/exam1.png -o ./output/
```

## Output Structure

The offline pipeline creates the following directory structure:

```
output_folder/
├── dewarped/           # Preprocessed, cleaned worksheet images
│   ├── image1_preprocessed.jpg
│   ├── image2_preprocessed.jpg
│   └── ...
├── debug/              # Debug images with detection overlays and tag information
│   ├── image1_debug.jpg
│   ├── image2_debug.jpg
│   └── ...
├── checked/            # Graded images with answer marks and score circles
│   ├── image1_checked.jpg
│   ├── image2_checked.jpg
│   └── ...
└── results.json        # Summary of all processing results
```

### results.json Format

Each entry contains:

```json
{
  "success": true,
  "input_file": "path/to/image.jpg",
  "worksheet_id": 2,
  "score": 19,
  "total_questions": 20,
  "answers": ["C", "A", "D", ...],
  "question_scores": [1, 1, 1, ...],
  "output_files": {
    "dewarped": "output/dewarped/image_preprocessed.jpg",
    "debug": "output/debug/image_debug.jpg",
    "checked": "output/checked/image_checked.jpg"
  }
}
```

## Processing Steps

The pipeline performs the following steps for each image:

### Step 1: Image Scanning
- Detects corner AprilTags (36h11 family) to locate worksheet boundaries
- Dewarps the image based on corner tag positions
- Identifies worksheet ID from orientation tag
- Cleans the document for better processing

✅ **Output**: Dewarped image

### Step 2: Preprocessing
- Applies document cleaning filters
- Enhances contrast and removes shadows
- Prepares image for OMR detection

✅ **Output**: Saved as `*_preprocessed.jpg`

### Step 3: OMR Processing
- Detects question row tags (25h9 family)
- Extracts regions of interest (ROIs) for each question
- Detects bubble marks in each ROI
- Analyzes fill ratio and circularity
- Grades answers against stored answer key

✅ **Output**: Answer detection, score calculation

### Step 4: Debug Image Generation
- Draws detected ROIs on the image
- Shows contour detections in green
- Highlights question areas for visualization

✅ **Output**: Saved as `*_debug.jpg`

### Step 5: Checked Image Generation
- Creates a marked-up version of the worksheet
- Shows correct answers with green checkmarks (✔)
- Shows incorrect answers with red X marks (✘)
- Displays score circle at top-left corner

✅ **Output**: Saved as `*_checked.jpg`

## Understanding the Output

### Debug Image
The debug image helps you verify that the pipeline correctly:
- Detected all corner tags
- Properly dewarped the image
- Located all question rows
- Extracted correct ROIs for each question

**What to look for:**
- ROI rectangles should align with question choices
- All 20 questions should have 2 ROIs (left and right columns)
- Contours in red show detected edges

### Checked Image
The checked image shows the final grading result with:
- **Green checkmarks (✔)** = Correct answers
- **Red X marks (✘)** = Incorrect answers
- **Score circle** = Total score displayed at top-left

**Note**: If no mark is detected for a question, it will appear blank/empty.

### Preprocessed Image
The preprocessed image shows the cleaned and dewarped worksheet:
- Removed perspective distortion
- Enhanced for better mark detection
- Ready for OMR analysis

## Logging Output

The pipeline provides detailed logging at each step. With verbose mode (`--verbose`), you get debug-level logs including:
- Detected tag information
- ROI coordinates for each question
- Bubble detection statistics (fill ratio, circularity, area)
- File paths and processing status

Example log output:
```
2026-02-06 19:04:10,535 - __main__ - INFO - ✓ Image scan successful. Worksheet ID: 2
2026-02-06 19:04:10,535 - __main__ - INFO - ✓ OMR processing successful. Score: 19/20
2026-02-06 19:04:10,549 - __main__ - INFO - ✓ Debug image saved: thick5_debug.jpg
```

## Troubleshooting

### Issue: "Less/more than 4 corner tags found"

**Cause**: The image doesn't have exactly 4 AprilTag corner markers, or they're not clearly visible.

**Solutions:**
- Ensure the entire worksheet is visible in the photo
- Check that corner tags are not damaged, folded, or obscured
- Improve lighting to make tags more visible
- Try taking a new photo with better angle and focus

### Issue: "Missing or invalid question tags"

**Cause**: Question row tags (25h9 family) could not be detected or are outside valid range.

**Solutions:**
- Verify all question row tags are visible
- Check that tags are clearly printed and not faded
- Ensure proper lighting and image focus
- Verify the worksheet template has correct tag configuration

### Issue: "No such file or directory" error

**Cause**: Output directory path doesn't exist or is inaccessible.

**Solutions:**
- Ensure the output directory path is correct
- Check file permissions (write access required)
- Create parent directories if needed: `mkdir -p ./output/`
- Use absolute paths if relative paths aren't working

### Issue: Answers detected incorrectly

**Possible causes:**
- Image quality (blur, poor lighting, shadows)
- Bubbles not properly filled or over-filled
- Worksheet template mismatch
- Ink color too light or too dark

**Solutions:**
- Use `--verbose` flag to see bubble detection details (fill ratio, circularity)
- Check the debug image to see detected bubbles and ROIs
- Verify the correct worksheet template is being used
- Improve photo quality and lighting
- Ensure marks are clearly filled within bubble boundaries

### Issue: Script runs but produces no output files

**Possible causes:**
- Image format not supported
- Image file corrupted
- Insufficient permissions to create output files

**Solutions:**
- Verify the image file is not corrupted: `file image.jpg`
- Check file permissions: `chmod 755 output_folder/`
- Ensure input image is in a supported format (JPG, PNG, BMP, TIFF)
- Use `--verbose` flag to see detailed error messages

## Supported Image Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **Bitmap** (.bmp)
- **TIFF** (.tiff, .tif)

## Performance

Processing time depends on image quality and system specifications:

- **Single image**: ~1-2 seconds
- **Batch of 10 images**: ~10-20 seconds
- **Batch of 100 images**: ~100-200 seconds

**Factors affecting performance:**
- Image resolution
- Number of detected contours
- System processor speed
- Disk I/O speed

## Advanced Usage

### Processing with Custom Output Paths

You can process multiple batches to different output directories:

```bash
# Process exam set 1
python offline_pipeline.py -i ./exam_set1/ -o ./results/exam_set1/

# Process exam set 2
python offline_pipeline.py -i ./exam_set2/ -o ./results/exam_set2/
```

### Analyzing Results Programmatically

Read and analyze the results.json file:

```python
import json

with open('results.json', 'r') as f:
    results = json.load(f)

for result in results:
    if result['success']:
        print(f"Score: {result['score']}/{result['total_questions']}")
        print(f"Answers: {result['answers']}")
```

## Integration with Main Service

The offline pipeline uses the same core processing functions as the main WhatsApp message service:
- `scan_image()` - Image dewarping and preprocessing
- `check_worksheet()` - OMR detection and grading
- `save_preprocessed()`, `save_debug()`, `save_checked()` - Image saving