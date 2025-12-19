#!/usr/bin/env python3
"""
Script to process logo images:
- Add white background if transparent
- Scale to square size maintaining aspect ratio
- Apply rounded corners
"""

from pathlib import Path

from PIL import Image, ImageDraw

# Configuration
LOGO_DIR = Path(__file__).parent
OUTPUT_SIZE = 512  # Final square size in pixels
CORNER_RADIUS = 50  # Radius for rounded corners
OUTPUT_SUFFIX = "_processed"  # Suffix for processed files

# Files to exclude from processing
EXCLUDE_FILES = ["model_comparison.png"]


def add_rounded_corners(image, radius):
    """
    Apply rounded corners to an image.

    Args:
        image: PIL Image object (RGBA)
        radius: Corner radius in pixels

    Returns:
        PIL Image with rounded corners
    """
    # Create a mask with rounded corners
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw a rounded rectangle
    draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)

    # Apply the mask
    output = Image.new("RGBA", image.size, (255, 255, 255, 0))
    output.paste(image, (0, 0))
    output.putalpha(mask)

    return output


def add_white_background(image):
    """
    Add white background to image if it has transparency.

    Args:
        image: PIL Image object

    Returns:
        PIL Image with white background (RGB mode)
    """
    # Convert to RGBA if not already
    if image.mode != "RGBA":
        if image.mode == "RGB":
            # Already has no transparency
            return image
        else:
            image = image.convert("RGBA")

    # Create white background
    background = Image.new("RGB", image.size, (255, 255, 255))

    # Paste image onto white background using alpha channel as mask
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel

    return background


def scale_to_square(image, size, maintain_aspect=True):
    """
    Scale image to fit in a square while maintaining aspect ratio.
    Centers the image on a white background.

    Args:
        image: PIL Image object
        size: Target square size
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        PIL Image scaled to square
    """
    if not maintain_aspect:
        return image.resize((size, size), Image.LANCZOS)

    # Calculate scaling factor to fit within square
    scale = min(size / image.width, size / image.height)
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)

    # Resize image
    resized = image.resize((new_width, new_height), Image.LANCZOS)

    # Create white square background
    square = Image.new("RGB", (size, size), (255, 255, 255))

    # Calculate position to center the image
    x = (size - new_width) // 2
    y = (size - new_height) // 2

    # Paste resized image onto center of square
    square.paste(resized, (x, y))

    return square


def process_logo(input_path, output_path):
    """
    Process a single logo image.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
    """
    print(f"Processing: {input_path.name}")

    # Load image
    image = Image.open(input_path)

    # Step 1: Add white background if needed
    image = add_white_background(image)

    # Step 2: Scale to square maintaining aspect ratio
    image = scale_to_square(image, OUTPUT_SIZE, maintain_aspect=True)

    # Step 3: Convert to RGBA for rounded corners
    image = image.convert("RGBA")

    # Step 4: Apply rounded corners (this creates transparency in corners)
    image = add_rounded_corners(image, CORNER_RADIUS)

    # Force output to PNG to preserve transparency in rounded corners
    output_path = output_path.with_suffix(".png")

    # Save processed image with transparency (RGBA mode) to show rounded corners
    image.save(output_path, quality=95)
    print(f"  → Saved: {output_path.name}")


def main():
    """Main function to process all logos in the directory."""
    print("Logo Processor")
    print("=" * 50)
    print(f"Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}px")
    print(f"Corner radius: {CORNER_RADIUS}px")
    print()

    # Find all image files
    image_extensions = [".png", ".jpg", ".jpeg"]
    logo_files = []

    for ext in image_extensions:
        logo_files.extend(LOGO_DIR.glob(f"*{ext}"))

    # Filter out excluded files and already processed files
    logo_files = [
        f for f in logo_files if f.name not in EXCLUDE_FILES and OUTPUT_SUFFIX not in f.stem
    ]

    if not logo_files:
        print("No logo files found to process.")
        return

    print(f"Found {len(logo_files)} logo(s) to process:\n")

    # Process each logo
    for logo_path in sorted(logo_files):
        # Generate output filename
        output_name = f"{logo_path.stem}{OUTPUT_SUFFIX}{logo_path.suffix}"
        output_path = LOGO_DIR / output_name

        try:
            process_logo(logo_path, output_path)
        except Exception as e:
            print(f"  ✗ Error processing {logo_path.name}: {e}")

    print()
    print("=" * 50)
    print("Processing complete!")


if __name__ == "__main__":
    main()
