#!/usr/bin/env python3
"""
Offline Pipeline - Command-line interface for processing worksheets offline.

This module provides a CLI tool to process worksheet images locally without 
the WhatsApp messaging infrastructure. It performs the same OMR detection and 
grading as the message_service, saving dewarped, debug, and checked images 
to specified output folders.

Usage:
    python offline_pipeline.py --input <path_to_image> --output <output_folder>
    
Example:
    python offline_pipeline.py --input ./test_image.jpg --output ./results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

from services.image_service import scan_image, save_preprocessed, save_debug, save_checked
from services.grading_service import check_worksheet
from models import InputImageMeta, WorksheetTemplate
from config import SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_output_directories(output_folder: Path) -> Tuple[Path, Path, Path, Path, Path]:
    """Create output subdirectories for dewarped, debug, checked, cropped, and bubble images.
    
    Args:
        output_folder (Path): Root output folder
        
    Returns:
        Tuple of (dewarped_dir, debug_dir, checked_dir, cropped_dir, bubbles_dir)
    """
    dewarped_dir = output_folder / "dewarped"
    debug_dir = output_folder / "debug"
    checked_dir = output_folder / "checked"
    cropped_dir = output_folder / "cropped"
    bubbles_dir = output_folder / "bubbles"
    
    for directory in [dewarped_dir, debug_dir, checked_dir, cropped_dir, bubbles_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", directory)
    
    return dewarped_dir, debug_dir, checked_dir, cropped_dir, bubbles_dir

def process_worksheet(
    input_path: Path,
    dewarped_dir: Path,
    debug_dir: Path,
    checked_dir: Path,
    cropped_dir: Path,
    bubbles_dir: Path
) -> Optional[dict]:
    """Process a single worksheet image through the OMR pipeline.
    
    Args:
        input_path (Path): Path to the input worksheet image
        dewarped_dir (Path): Directory to save dewarped (preprocessed) images
        debug_dir (Path): Directory to save debug images with detections
        checked_dir (Path): Directory to save checked (graded) images
        cropped_dir (Path): Directory to save cropped images
        bubbles_dir (Path): Directory to save individual bubble and ROI images
        
    Returns:
        dict: Results containing answers, scores, and paths to saved images
        None: If processing failed at any stage
    """
    try:
        logger.info("Processing image: %s", input_path)
        
        # Validate input file exists
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            return None
        
        # Initialize input image metadata
        input_image = InputImageMeta(image_path=str(input_path))
        
        # Step 1: Scan image (detect corner tags, dewarp, clean)
        logger.info("Step 1: Scanning image and detecting corner tags...")
        try:
            worksheet = scan_image(input_image)
            logger.info("✓ Image scan successful. Worksheet ID: %s", worksheet.worksheet_id)
        except ValueError as e:
            logger.error("✗ Image scan failed: %s", e)
            logger.error("  Possible causes: Missing/incorrect corner tags, poor image quality")
            return None
        except Exception as e:
            logger.exception("✗ Unexpected error during image scan: %s", e)
            return None
        
        # Step 2: Temporarily override output directories for save functions
        original_dewarped = SETTINGS.DEWARPED_PATH
        original_debug = SETTINGS.DEBUG_PATH
        original_checked = SETTINGS.CHECKED_PATH
        
        SETTINGS.DEWARPED_PATH = str(dewarped_dir)
        SETTINGS.DEBUG_PATH = str(debug_dir)
        SETTINGS.CHECKED_PATH = str(checked_dir)
        CROPPED_PATH = str(cropped_dir)
        
        # Override BUBBLES_FOLDER for this processing run
        from services import grading_service
        original_bubbles_folder = grading_service.BUBBLES_FOLDER
        grading_service.BUBBLES_FOLDER = bubbles_dir
        
        try:
            # save cropped image
            cropped_filename = f"{input_path.stem}_cropped.jpg"
            cropped_path = cropped_dir / cropped_filename
            cropped_image = worksheet.cropped_image
            cropped_image.save(cropped_path)
            logger.info("✓ Cropped image saved: %s", cropped_filename)

            # Step 3: Save preprocessed (dewarped) image
            logger.info("Step 2: Saving preprocessed image...")
            save_preprocessed(worksheet)
            dewarped_filename = f"{input_path.stem}_preprocessed.jpg"
            logger.info("✓ Preprocessed image saved: %s", dewarped_filename)
            
            # Step 4: Process OMR answers
            logger.info("Step 3: Processing OMR answers...")
            answers, q_score, omr_success = check_worksheet(worksheet, use_classifier=True, debug=True)
            logging.info("✓ OMR processing completed. Detected answers: %s", answers)
            
            if not omr_success:
                logger.warning("✗ OMR processing failed: Missing or invalid question tags")
                logger.warning("  Check that all question row tags (25h9) are visible and properly printed")
                return None
            
            score = sum(q_score) if q_score else 0
            logger.info("✓ OMR processing successful. Score: %d/%d", score, len(answers))
            
            # Step 5: Save debug image
            logger.info("Step 4: Saving debug image with detections...")
            save_debug(worksheet)
            debug_filename = f"{input_path.stem}_debug.jpg"
            logger.info("✓ Debug image saved: %s", debug_filename)
            
            # Step 6: Save checked image (with marks)
            logger.info("Step 5: Saving checked image with marks...")
            save_checked(worksheet)
            checked_filename = f"{input_path.stem}_checked.jpg"
            logger.info("✓ Checked image saved: %s", checked_filename)
            
            # Step 7: Compile results
            results = {
                "success": True,
                "input_file": str(input_path),
                "worksheet_id": worksheet.worksheet_id,
                "score": score,
                "total_questions": len(answers),
                "answers": answers,
                "question_scores": q_score,
                "output_files": {
                    "dewarped": str(dewarped_dir / dewarped_filename),
                    "debug": str(debug_dir / debug_filename),
                    "checked": str(checked_dir / checked_filename),
                    "cropped": str(cropped_dir / cropped_filename)
                }
            }
            
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 60)
            logger.info("Score: %d/%d", score, len(answers))
            logger.info("Answers: %s", answers)
            logger.info("Output files:")
            logger.info("  - Dewarped: %s", dewarped_dir / dewarped_filename)
            logger.info("  - Debug:    %s", debug_dir / debug_filename)
            logger.info("  - Checked:  %s", checked_dir / checked_filename)
            logger.info("=" * 60)
            
            return results
            
        finally:
            # Restore original settings
            SETTINGS.DEWARPED_PATH = original_dewarped
            SETTINGS.DEBUG_PATH = original_debug
            SETTINGS.CHECKED_PATH = original_checked
            grading_service.BUBBLES_FOLDER = original_bubbles_folder
    
    except Exception as e:
        logger.exception("Unexpected error during processing: %s", e)
        return None


def main():
    """Main entry point for the offline pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Process worksheet images offline with OMR detection and grading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python offline_pipeline.py --input worksheet.jpg --output ./results/
  
  # Process multiple images in a folder
  python offline_pipeline.py --input images/ --output ./results/
  
  # Verbose output
  python offline_pipeline.py --input worksheet.jpg --output ./results/ --verbose
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image file or folder containing images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output folder where dewarped, debug, and checked images will be saved'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Create output directories
    try:
        dewarped_dir, debug_dir, checked_dir, cropped_dir, bubbles_dir = create_output_directories(output_path)
    except Exception as e:
        logger.error("Failed to create output directories: %s", e)
        return 1
    
    # Process image(s)
    results_list = []
    
    if input_path.is_file():
        # Single image file
        logger.info("Processing single image file: %s", input_path)
        result = process_worksheet(input_path, dewarped_dir, debug_dir, checked_dir, cropped_dir, bubbles_dir)
        
        if result:
            results_list.append(result)
        else:
            logger.error("Failed to process image: %s", input_path)
            return 1
    
    elif input_path.is_dir():
        # Multiple images in folder
        logger.info("Processing images from folder: %s", input_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.error("No image files found in folder: %s", input_path)
            return 1
        
        logger.info("Found %d image file(s)", len(image_files))
        
        for idx, image_file in enumerate(image_files, 1):
            logger.info("\n[%d/%d] Processing: %s", idx, len(image_files), image_file.name)
            result = process_worksheet(image_file, dewarped_dir, debug_dir, checked_dir, cropped_dir, bubbles_dir)
            
            if result:
                results_list.append(result)
            else:
                logger.warning("Failed to process: %s", image_file.name)
    
    else:
        logger.error("Input path does not exist: %s", input_path)
        return 1
    
    # Save results summary
    if results_list:
        results_file = output_path / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        logger.info("\nResults summary saved to: %s", results_file)
        
        successful = sum(1 for r in results_list if r.get('success', False))
        logger.info("Processing summary: %d/%d successful", successful, len(results_list))
        
        return 0
    else:
        logger.error("No images were successfully processed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
