#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  BatchGen Checkpoint Converter CLI                                            #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Convert HuggingFace model checkpoints to BatchGen format for optimized       #
#  SSD read performance during inference.                                       #
# ---------------------------------------------------------------------------- #
"""
CLI tool for converting HuggingFace checkpoints to BatchGen format.

This tool converts .safetensors and .pt checkpoint files to BatchGen's
optimized format (.bin + .json metadata files) for peak SSD read performance.

Usage:
    python -m batchgen.tools.convert_checkpoint --input-dir /path/to/model
    python -m batchgen.tools.convert_checkpoint -i /path/to/model -o /path/to/output
    python -m batchgen.tools.convert_checkpoint -i /path/to/model --force
    python -m batchgen.tools.convert_checkpoint -i /path/to/model --validate-only
"""

import argparse
import logging
import sys


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI tool."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [BatchGen Converter] - %(levelname)s - %(message)s',
    )


def main(args=None):
    """Main entry point for the checkpoint converter CLI."""
    parser = argparse.ArgumentParser(
        prog='batchgen-convert-checkpoint',
        description='Convert HuggingFace model checkpoints to BatchGen format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a model directory (output to <input-dir>/converted_ckpt)
  python -m batchgen.tools.convert_checkpoint --input-dir /models/DeepSeek-V3

  # Specify custom output directory
  python -m batchgen.tools.convert_checkpoint -i /models/DeepSeek-V3 -o /converted/DeepSeek-V3

  # Force reconversion even if already converted
  python -m batchgen.tools.convert_checkpoint -i /models/DeepSeek-V3 --force

  # Only validate existing converted files
  python -m batchgen.tools.convert_checkpoint -i /models/DeepSeek-V3 --validate-only

Output Format:
  The converter creates .bin and .json file pairs for each checkpoint shard:
  - .bin: Raw contiguous tensor bytes for optimal SSD read performance
  - .json: Metadata with tensor names, dtypes, shapes, offsets, and byte sizes
        """
    )

    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing .safetensors or .pt checkpoint files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for converted files (default: <input-dir>/converted_ckpt)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reconversion even if output directory already exists with valid files'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing converted files without converting'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(parsed_args.verbose)

    # Import here to avoid import errors if dependencies are missing
    try:
        from batchgen.ckpt_converter.ckpt_converter import ckpt_converter
    except ImportError as e:
        logging.error(f"Failed to import ckpt_converter: {e}")
        logging.error("Make sure BatchGen is properly installed.")
        return 1

    converter = ckpt_converter()
    input_dir = parsed_args.input_dir
    output_dir = parsed_args.output_dir

    # Default output directory
    if output_dir is None:
        import os
        output_dir = os.path.join(input_dir, "converted_ckpt")

    try:
        if parsed_args.validate_only:
            # Validation-only mode
            import os
            if not os.path.exists(output_dir):
                logging.error(f"Output directory {output_dir} does not exist. Nothing to validate.")
                return 1

            logging.info(f"Validating converted files in {output_dir}...")
            is_valid, error_msg = converter.validate_converted_directory(input_dir, output_dir)

            if is_valid:
                logging.info("Validation successful! All converted files are consistent with source checkpoints.")
                return 0
            else:
                logging.error(f"Validation failed: {error_msg}")
                return 1
        else:
            # Conversion mode
            logging.info(f"Starting checkpoint conversion...")
            logging.info(f"  Input directory:  {input_dir}")
            logging.info(f"  Output directory: {output_dir}")
            logging.info(f"  Force reconvert:  {parsed_args.force}")

            result_dir = converter.convert_model_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                force=parsed_args.force
            )

            logging.info(f"Conversion completed successfully!")
            logging.info(f"Converted files are in: {result_dir}")
            logging.info("")
            logging.info("To use with BatchGen server, provide this path as --cache-dir:")
            logging.info(f"  batchgen-server --cache-dir {input_dir}")
            return 0

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
        return 1
    except RuntimeError as e:
        logging.error(f"Conversion failed: {e}")
        return 1
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
