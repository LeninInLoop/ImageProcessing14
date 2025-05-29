import os
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd


class ImageUtils:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError
        image = Image.open(image_path)
        return np.array(image).astype(np.float64)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    @staticmethod
    def perform_thresholding(image: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(image > threshold, 1, 0).astype(np.bool_)

    @staticmethod
    def convert_to_gray_scale(image: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(image.astype(np.uint8)).convert("L"))

    @staticmethod
    def save_image(image_path: str, image: np.ndarray) -> None:
        return Image.fromarray(image.astype(np.uint8)).save(image_path)

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return np.array(Image.fromarray(image.astype(np.uint8)).resize(size))


class Helper:
    @staticmethod
    def create_directories(directories: Dict) -> None:
        for directory in directories.values():
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def find_first_coordinate(image: np.ndarray) -> Tuple[int, int]:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j]:
                    return i, j
        return None, None

    @staticmethod
    def has_foreground_pixels(image: np.ndarray) -> bool:
        """Check if the image has any foreground pixels"""
        return np.any(image)

    @staticmethod
    def perform_dilation(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
        image = image.astype(np.bool_)
        structuring_element = structuring_element.astype(np.bool_)

        img_h, img_w = image.shape
        se_h, se_w = structuring_element.shape

        pad_h = se_h // 2
        pad_w = se_w // 2

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=False)
        result = np.zeros_like(image, dtype=np.bool_)

        # Perform dilation
        for i in range(img_h):
            for j in range(img_w):
                roi = padded_image[i:i + se_h, j:j + se_w]
                if np.any(roi & structuring_element):
                    result[i, j] = True
        return result

    @staticmethod
    def perform_intersection(image: np.ndarray, dilated_image: np.ndarray) -> np.ndarray:
        return image & dilated_image

    @staticmethod
    def extract_single_shape(binary_image: np.ndarray, start_coordinate: Tuple[int, int]) -> np.ndarray:
        """Extract a single connected shape using iterative dilation"""
        if start_coordinate[0] is None or start_coordinate[1] is None:
            return np.zeros_like(binary_image, dtype=np.bool_)

        # Create initial image with single pixel
        current_shape = np.zeros(binary_image.shape, dtype=np.bool_)
        current_shape[start_coordinate[0], start_coordinate[1]] = True

        structuring_element = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])

        while True:
            # Dilate current shape
            dilated_shape = Helper.perform_dilation(
                image=current_shape,
                structuring_element=structuring_element
            )

            # Intersect with the original binary image
            intersected_shape = Helper.perform_intersection(
                image=binary_image,
                dilated_image=dilated_shape
            )

            # Check if shape has stopped growing
            if np.array_equal(intersected_shape, current_shape):
                break

            current_shape = intersected_shape.copy()

        return current_shape

    @staticmethod
    def remove_shape_from_image(binary_image: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """Remove extracted shape from the binary image"""
        return binary_image & (~shape)

    @staticmethod
    def get_shape_coordinates(shape: np.ndarray) -> List[Tuple[int, int]]:
        """Get all coordinates where shape pixels are True"""
        coordinates = []
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]:
                    coordinates.append((i, j))
        return coordinates

    @staticmethod
    def extract_all_shapes(binary_image: np.ndarray) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        """Extract all shapes from binary image"""
        shapes = []
        coordinates_list = []
        working_image = binary_image.copy()
        shape_count = 0

        print("Starting shape extraction...")

        while Helper.has_foreground_pixels(working_image):
            # Find first coordinate in the remaining image
            coordinate = Helper.find_first_coordinate(working_image)

            if coordinate[0] is None:
                break

            print(f"Extracting shape {shape_count + 1} starting at coordinate: {coordinate}")

            # Extract single shape
            extracted_shape = Helper.extract_single_shape(working_image, coordinate)

            # Store shape and its coordinates
            shapes.append(extracted_shape)
            shape_coordinates = Helper.get_shape_coordinates(extracted_shape)
            coordinates_list.append(shape_coordinates)

            # Remove extracted shape from working image
            working_image = Helper.remove_shape_from_image(working_image, extracted_shape)

            shape_count += 1
            print(f"Shape {shape_count} extracted with {len(shape_coordinates)} pixels")

        print(f"Total shapes extracted: {len(shapes)}")
        return shapes, coordinates_list

    @staticmethod
    def create_colored_visualization(shapes: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a colored visualization of all shapes"""
        # Create RGB image
        colored_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        # Generate distinct colors for each shape
        colors = plt.cm.Set3(np.linspace(0, 1, len(shapes)))  # Use colormap for distinct colors

        for i, shape in enumerate(shapes):
            # Convert color to 0-255 range
            color = (colors[i][:3] * 255).astype(np.uint8)

            # Apply color to shape pixels
            mask = shape.astype(np.bool_)
            colored_image[mask] = color

        return colored_image

    @staticmethod
    def save_individual_shapes(shapes: List[np.ndarray], output_directory: str):
        """Save each shape individually"""
        for i, shape in enumerate(shapes):
            # Convert boolean to uint8 (0 or 255)
            shape_image = (shape * 255).astype(np.uint8)
            filename = os.path.join(output_directory, f"shape_{i + 1:03d}.png")
            ImageUtils.save_image(filename, shape_image)
            print(f"Saved individual shape: {filename}")

    @staticmethod
    def save_coordinates_to_csv(coordinates_list: List[List[Tuple[int, int]]], output_path: str):
        """Save all shape coordinates to a CSV file with a column for each shape"""
        if not coordinates_list:
            print("No coordinates to save!")
            return

        # Find the maximum number of coordinates in any shape
        max_coords = max(len(coords) for coords in coordinates_list)

        # Create data dictionary for DataFrame
        data = {}

        # Add coordinates for each shape
        for i, coords in enumerate(coordinates_list):
            shape_name = f"Shape_{i + 1}"

            # Convert coordinates to strings in format "(row,col)"
            coord_strings = [f"({coord[0]},{coord[1]})" for coord in coords]

            # Pad with empty strings if this shape has fewer coordinates
            while len(coord_strings) < max_coords:
                coord_strings.append("")

            data[shape_name] = coord_strings

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"Coordinates saved to CSV: {output_path}")
        print(f"CSV contains {len(coordinates_list)} shape columns with up to {max_coords} coordinate rows each")

    @staticmethod
    def save_coordinates_to_csv_separate_columns(coordinates_list: List[List[Tuple[int, int]]], output_path: str):
        """Save coordinates to CSV with separate Row and Column columns for each shape"""
        if not coordinates_list:
            print("No coordinates to save!")
            return

        # Find the maximum number of coordinates in any shape
        max_coords = max(len(coords) for coords in coordinates_list)

        # Create data dictionary for DataFrame
        data = {}

        # Add coordinates for each shape with separate row and column columns
        for i, coords in enumerate(coordinates_list):
            shape_name = f"Shape_{i + 1}"

            # Separate rows and columns
            rows = [coord[0] for coord in coords]
            cols = [coord[1] for coord in coords]

            # Pad with NaN if this shape has fewer coordinates
            while len(rows) < max_coords:
                rows.append(np.nan)
                cols.append(np.nan)

            data[f"{shape_name}_Row"] = rows
            data[f"{shape_name}_Col"] = cols

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"Coordinates saved to CSV (separate columns): {output_path}")
        print(
            f"CSV contains {len(coordinates_list)} shapes with separate Row/Col columns, up to {max_coords} coordinate rows each")


def main():
    directories = {
        "base_path": "Images",
        "gray_scale": os.path.join("Images", "GrayScale"),
        "results": os.path.join("Images", "Results"),
        "individual_shapes": os.path.join("Images", "IndividualShapes"),
    }
    Helper.create_directories(directories)

    # ================================================================================
    # Load Original Image
    # ================================================================================
    original_image_path = os.path.join("Images", "original.bmp")
    original_image = ImageUtils.load_image(original_image_path)
    print("Original image shape: ", original_image.shape)

    # resized_original_image = ImageUtils.resize_image(original_image, size=(512, 512))
    # ImageUtils.save_image(
    #     image_path=original_image_path,
    #     image=resized_original_image
    # )
    # raise SystemExit

    # ================================================================================
    # Convert to Gray Scale
    # ================================================================================
    gray_scaled_image = ImageUtils.convert_to_gray_scale(original_image)
    ImageUtils.save_image(
        image_path=os.path.join(directories["gray_scale"], "gray_scale_original.png"),
        image=gray_scaled_image
    )

    # ================================================================================
    # Perform Thresholding
    # ================================================================================
    binary_image = ImageUtils.perform_thresholding(
        image=gray_scaled_image,
        threshold=np.mean(gray_scaled_image)
    )
    binary_image = binary_image.astype(np.bool_)

    ImageUtils.save_image(
        image_path=os.path.join(directories["gray_scale"], "binary_original.png"),
        image=(binary_image * 255).astype(np.uint8)
    )

    # ===============================================================================
    # Extract All Shapes
    # ===============================================================================
    print("\n" + "=" * 50)
    print("EXTRACTING ALL SHAPES")
    print("=" * 50)

    shapes, coordinates_list = Helper.extract_all_shapes(binary_image)

    # Print summary of extracted shapes
    print(f"\nExtraction Summary:")
    print(f"Total shapes found: {len(shapes)}")
    for i, coords in enumerate(coordinates_list):
        print(f"Shape {i + 1}: {len(coords)} pixels")

    # ===============================================================================
    # Save Coordinates to CSV
    # ===============================================================================
    if coordinates_list:
        print("\n" + "=" * 50)
        print("SAVING COORDINATES TO CSV")
        print("=" * 50)

        # Save coordinates in single format (row,col) per cell
        csv_output_path = os.path.join(directories["results"], "shape_coordinates.csv")
        Helper.save_coordinates_to_csv(coordinates_list, csv_output_path)

        # Save coordinates in separate row/column format
        csv_separate_path = os.path.join(directories["results"], "shape_coordinates_separate.csv")
        Helper.save_coordinates_to_csv_separate_columns(coordinates_list, csv_separate_path)

    # ===============================================================================
    # Create Colored Visualization
    # ===============================================================================
    if shapes:
        print("\nCreating colored visualization...")
        colored_image = Helper.create_colored_visualization(shapes, binary_image.shape)

        # Save colored visualization
        colored_output_path = os.path.join(directories["results"], "all_shapes_colored.png")
        ImageUtils.save_image(colored_output_path, colored_image)
        print(f"Colored visualization saved: {colored_output_path}")

        # ===============================================================================
        # Save Individual Shapes
        # ===============================================================================
        print("\nSaving individual shapes...")
        Helper.save_individual_shapes(shapes, directories["individual_shapes"])

        # ===============================================================================
        # Display Results using Matplotlib
        # ===============================================================================
        print("\nDisplaying results...")

        # Calculate grid size for displaying all shapes
        num_shapes = len(shapes)

        # Create the main figure for overview
        fig_overview, axes_overview = plt.subplots(1, 2, figsize=(15, 6))
        fig_overview.suptitle('Shape Extraction Overview', fontsize=16)

        # Original binary image
        axes_overview[0].imshow(binary_image, cmap='gray')
        axes_overview[0].set_title('Original Binary Image')
        axes_overview[0].axis('off')

        # Colored visualization
        axes_overview[1].imshow(colored_image)
        axes_overview[1].set_title(f'All Shapes Colored ({num_shapes} shapes)')
        axes_overview[1].axis('off')

        plt.tight_layout()

        # Save overview figure
        overview_output_path = os.path.join(directories["results"], "extraction_overview.png")
        plt.savefig(overview_output_path, dpi=300, bbox_inches='tight')
        print(f"Overview plot saved: {overview_output_path}")

        # Create a separate figure for all individual shapes
        if num_shapes > 0:
            # Calculate grid dimensions for individual shapes
            cols = min(4, num_shapes)  # Max 4 columns
            rows = (num_shapes + cols - 1) // cols  # Calculate required rows

            fig_shapes, axes_shapes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            fig_shapes.suptitle(f'All {num_shapes} Individual Shapes', fontsize=16)

            # Handle single row case
            if rows == 1:
                if cols == 1:
                    axes_shapes = [axes_shapes]
                else:
                    axes_shapes = [axes_shapes]

            # Flatten axes for easier indexing
            if rows > 1:
                axes_flat = axes_shapes.flatten()
            else:
                axes_flat = axes_shapes[0] if cols > 1 else [axes_shapes]

            # Display each shape
            for i in range(num_shapes):
                if rows == 1 and cols == 1:
                    ax = axes_shapes
                elif rows == 1:
                    ax = axes_shapes[i]
                else:
                    ax = axes_flat[i]

                ax.imshow(shapes[i], cmap='gray')
                ax.set_title(f'Shape {i + 1}\n({len(coordinates_list[i])} pixels)')
                ax.axis('off')

            # Hide empty subplots
            total_subplots = rows * cols
            for i in range(num_shapes, total_subplots):
                if rows == 1:
                    if cols > 1:
                        axes_shapes[i].axis('off')
                else:
                    axes_flat[i].axis('off')

            plt.tight_layout()

            # Save individual shapes figure
            shapes_output_path = os.path.join(directories["results"], "all_individual_shapes.png")
            plt.savefig(shapes_output_path, dpi=300, bbox_inches='tight')
            print(f"Individual shapes plot saved: {shapes_output_path}")

        plt.show()

    else:
        print("No shapes found in the image!")

    print("\nShape extraction completed successfully!")


if __name__ == '__main__':
    main()