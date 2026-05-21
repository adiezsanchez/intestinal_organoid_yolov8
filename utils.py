import os
import shutil
import pandas as pd
import numpy as np
import math
import tifffile
from pathlib import Path

# On Windows + pixi, Pillow/skimage may fail to resolve native DLLs unless
# the environment's Library/bin directory is added to the DLL search path.
if os.name == "nt":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dll_dir = Path(conda_prefix) / "Library" / "bin"
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))

import pyclesperanto_prototype as cle  # version 0.24.1
import napari_segment_blobs_and_things_with_membranes as nsbatwm  # version 0.3.6
from tqdm import tqdm
from skimage import measure, io
from skimage.measure import regionprops
from skimage.color import rgb2gray
from apoc import ObjectSegmenter, ObjectClassifier

_IMAGE_SUFFIXES = {".tif", ".tiff", ".png"}
_RESULTS_SUBDIRS = {"per_organoid_stats", "summary_stats"}


def load_image(file_path):
    """Load a microscopy image from TIF or PNG."""
    file_path = Path(file_path)
    if file_path.suffix.lower() in (".tif", ".tiff"):
        img = tifffile.imread(str(file_path), is_ome=False)
    else:
        img = io.imread(str(file_path))
        # Pillow loads 16-bit PNG as int32; keep uint16 for downstream scaling.
        if img.dtype == np.int32:
            img = img.astype(np.uint16)
    return img


def to_uint8_rgb(image):
    """Convert grayscale or multi-channel image to contiguous uint8 RGB for YOLO/OpenCV."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.float64)
        img -= img.min()
        scale = img.max()
        if scale > 0:
            img = (img / scale * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    return np.ascontiguousarray(img)


def ensure_rgb(image):
    """Stack grayscale to 3 channels for RGB-only downstream steps (e.g. YOLO)."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[..., :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def read_images(directory_path):
    """Reads all the images in the input path and organizes them according to the well_id"""
    directory_path = Path(directory_path)
    images_per_well = {}

    eligible = []
    for file_path in directory_path.glob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        filename = file_path.stem
        if "_z" not in filename:
            continue
        eligible.append(file_path)

    # Plate_R is grayscale; prefer RGB-like channels (Plate_M, Plate_D, …) when present.
    use_plate_r = not any("Plate_R" not in p.stem for p in eligible)

    for file_path in eligible:
        filename = file_path.stem
        if "Plate_R" in filename and not use_plate_r:
            continue

        last_part = filename.split("_")[-1]
        well_id = last_part[:3]

        if well_id not in images_per_well:
            images_per_well[well_id] = []

        images_per_well[well_id].append(str(file_path))

    for well_id in images_per_well:
        images_per_well[well_id].sort()

    return images_per_well


def folder_has_z_stacks(directory_path):
    """Return True if the folder contains EVOS z-slice images (Plate_D PNG, Plate_M TIF, etc.)."""
    directory_path = Path(directory_path)
    for file_path in directory_path.glob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in _IMAGE_SUFFIXES
            and "_z" in file_path.stem
        ):
            return True
    return False


def list_plate_directories(data_folder, exclude_4x=True):
    """List plate scan folders under a DATA_FOLDER (or return the folder itself if it is a plate)."""
    data_folder = Path(data_folder)
    if not data_folder.exists():
        raise FileNotFoundError(f"DATA_FOLDER does not exist: {data_folder}")

    if folder_has_z_stacks(data_folder):
        return [data_folder]

    plate_dirs = []
    for subfolder in sorted(data_folder.iterdir()):
        if not subfolder.is_dir():
            continue
        if exclude_4x and "4X" in subfolder.name:
            continue
        if folder_has_z_stacks(subfolder):
            plate_dirs.append(subfolder)

    if not plate_dirs:
        raise FileNotFoundError(
            f"No plate folders with z-stack images found under: {data_folder}"
        )

    return plate_dirs


def min_intensity_projection(image_paths):
    """Takes a collection of image paths containing one z-stack per file and performs minimum intensity projection"""
    stack = np.stack(
        [ensure_rgb(load_image(p)).astype(np.float32) for p in image_paths], axis=0
    )
    min_proj = np.min(stack, axis=0)

    return to_uint8_rgb(min_proj)


def save_min_projection_imgs(images_per_well, output_dir="./output/MIN_projections"):
    """Takes a images_per_well from read_images as input, performs minimum intensity projection and saves the resulting image on a per well basis"""
    for well_id, files in images_per_well.items():
        if not files:
            continue
        # Perform minimum intensity projection of the stack stored under well_id key
        min_proj = min_intensity_projection(files)

        # Create a directory to store the tif files if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Construct the output file path
        output_path = os.path.join(output_dir, f"{well_id}.tif")

        # Save the resulting minimum projection
        tifffile.imwrite(output_path, min_proj)


def predict_masks(
    input_folder,
    model_dir="./models/GPU_1px_cle_eroded_labels.pt",
    conf_threshold=0.6,
    output_dir="./output/predictions",
):
    """Takes a directory containing minimum intensity projections as input and outputs the predicted masks burnt-in on top of the input images"""
    from ultralytics import YOLO

    # Define the directory containing your files
    directory_path = Path(input_folder)

    # Load the pre-trained model you want to apply
    model = YOLO(model_dir)

    # Loop through all input images (minimum intensity projections)
    for image_path in directory_path.glob("*.tif"):

        # Get the filename without the extension
        filename = image_path.stem

        image = to_uint8_rgb(load_image(image_path))
        results = model.predict(image, conf=conf_threshold)

        im_array = results[0].plot(conf=False, labels=False, boxes=True, line_width=2)

        # Create a directory to store the tif files if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Construct the output file path
        output_path = os.path.join(output_dir, f"{filename}.tif")

        # Save the resulting minimum projection
        tifffile.imwrite(output_path, im_array)


def extract_stats(
    input_folder,
    model_dir="./models/GPU_1px_cle_eroded_labels.pt",
    conf_threshold=0.6,
    output_dir="./output/predictions",
):
    """Takes a directory containing minimum intensity projections, performs inference and returns a df containing regionprops per object per class per well_id"""
    from ultralytics import YOLO

    # Define the directory containing your files
    directory_path = Path(input_folder)

    # Load the pre-trained model you want to apply
    model = YOLO(model_dir)

    # Initialize a list to store the properties for all images
    all_props_list = []

    # Loop through all input images (minimum intensity projections)
    for image_path in directory_path.glob("*.tif"):

        # Get the filename without the extension
        filename = image_path.stem

        image = to_uint8_rgb(load_image(image_path))
        results = model.predict(image, conf=conf_threshold)

        # Access the first position in the resulting YOLO results list
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            continue

        extracted_masks = result.masks.data

        # Extract the boxes, which contain class IDs
        detected_boxes = result.boxes.data

        # Extract class IDs from the detected boxes
        class_labels = detected_boxes[:, -1].int().tolist()

        # Initialize a dictionary to hold masks by class
        masks_by_class = {name: [] for name in result.names.values()}

        # Iterate through the masks and class labels
        for mask, class_id in zip(extracted_masks, class_labels):
            class_name = result.names[class_id]  # Map class ID to class name
            masks_by_class[class_name].append(mask.cpu().numpy())

        # for class_name, masks in masks_by_class.items():
        # print(f"Class Name: {class_name}, Number of Masks: {len(masks)}")

        # Initialize a list to store the properties
        props_list = []

        # Iterate through all classes
        for class_name, masks in masks_by_class.items():
            # Iterate through the masks for this class
            for mask in masks:
                # Convert the mask to an integer type if it's not already
                mask = mask.astype(int)

                # Apply regionprops to the mask
                props = regionprops(mask)

                # Extract the properties you want (e.g., area, perimeter) and add them to the list
                for prop in props:
                    area = prop.area
                    area_filled = prop.area_filled
                    perimeter = prop.perimeter
                    circularity = min(1, (4 * math.pi * prop.area) / prop.perimeter**2)
                    eccentricity = prop.eccentricity
                    label = prop.label
                    solidity = prop.solidity
                    # Add other properties as needed

                    # Append the properties and class name to the list
                    props_list.append(
                        {
                            "well_id": filename,
                            "Class Name": class_name,
                            "Area": area,
                            "Area_filled": area_filled,
                            "Perimeter": perimeter,
                            "Circularity": circularity,
                            "Eccentricity": eccentricity,
                            "Solidity": solidity,
                        }
                    )

        # Extend all_props_list with the properties from this image
        all_props_list.extend(props_list)

    # Convert the aggregated list of dictionaries to a DataFrame
    all_props_df = pd.DataFrame(all_props_list)

    # Now props_df contains the properties and class names for all regions
    return all_props_df


def copy_csv_results(results_directory):
    """Copy all .csv files from each plate folder into a per_organoid_stats folder under results_directory"""

    # Create an empty list to store the subdirectories within results_directory
    subdirectories = []

    # Iterate over subfolders in the results_directory and add them to subdirectories list
    for subfolder in results_directory.iterdir():
        if subfolder.is_dir() and subfolder.name not in _RESULTS_SUBDIRS:
            subdirectories.append(subfolder.name)

    # Create the destination folder to copy the .csv files contained in each subdir
    try:
        csv_results_path = os.path.join(results_directory, "per_organoid_stats")
        os.makedirs(csv_results_path)
    except FileExistsError:
        print(f"Directory already exists: {csv_results_path}")

    # Iterate over each subdirectory to scan and copy .csv files contained within
    for subdir in subdirectories:
        # Build the path to each of the subdirectories
        subdirectory_path = Path(results_directory) / subdir

        # Check if the subdirectory exists
        if subdirectory_path.exists():
            # Scan for .csv files in each subdirectory
            for file_path in subdirectory_path.glob("*.csv"):
                # Copy the file to the destination subfolder
                shutil.copy2(file_path, csv_results_path)


def extract_summary_stats(csv_path):
    """Processes a per_organoid .csv results file counting the number of occurrences and calculate the average of each property returning a summary_stats_df"""
    # Read the .csv into a pandas DataFrame
    df = pd.read_csv(csv_path)
    df["Class Name"] = df["Class Name"].astype(str).str.strip().str.lower()

    # Grouping and counting occurrences
    grouped_counts = df.groupby(["well_id", "Class Name"]).size().unstack(fill_value=0)

    # Specifying the columns to calculate the average for
    columns_to_average = [
        "Area",
        "Area_filled",
        "Perimeter",
        "Circularity",
        "Eccentricity",
        "Solidity",
    ]

    # Grouping by 'well_id' and 'Class Name' and calculating the mean
    average_values = (
        df.groupby(["well_id", "Class Name"])[columns_to_average].mean().reset_index()
    )

    # Joining the counts back to the original dataframe
    df_merged = average_values.merge(grouped_counts, on="well_id", how="left")

    # Optionally, if you want to create a specific column for each count based on the class in each row
    for class_name in grouped_counts.columns:
        df_merged[class_name] = df_merged[class_name].where(
            df_merged["Class Name"] == class_name, 0
        )

    # List of columns to update with their maximum value per well_id
    columns_to_maximize = ["dead", "differentiated", "undifferentiated"]

    # If a class is missing in a plate, create the count column with zeros
    # so downstream transforms/ratios remain stable.
    missing_class_columns = [
        column for column in columns_to_maximize if column not in df_merged.columns
    ]
    if missing_class_columns:
        print(
            f"Warning: missing classes in {Path(csv_path).name}: {missing_class_columns}. "
            "Defaulting counts to 0."
        )
        for column in missing_class_columns:
            df_merged[column] = 0

    # Apply transform to update each specified column with its max value per well_id
    for column in columns_to_maximize:
        df_merged[column] = df_merged.groupby("well_id")[column].transform("max")

    # Renaming the columns
    df_merged = df_merged.rename(
        columns={
            "dead": "nr_dead",
            "differentiated": "nr_organoids",
            "undifferentiated": "nr_spheroids",
        }
    )

    total_count = df_merged["nr_dead"] + df_merged["nr_organoids"] + df_merged["nr_spheroids"]
    live_count = df_merged["nr_organoids"] + df_merged["nr_spheroids"]

    # Safe divisions avoid NaN/inf when no objects are detected in a well.
    df_merged["dead_ratio"] = np.where(
        total_count > 0, df_merged["nr_dead"] / total_count, 0
    )
    df_merged["organoid_ratio"] = np.where(
        live_count > 0, df_merged["nr_organoids"] / live_count, 0
    )
    df_merged["spheroid_ratio"] = np.where(
        live_count > 0, df_merged["nr_spheroids"] / live_count, 0
    )

    # Extract the plate_name from the csv_path
    csv_path = Path(csv_path)
    plate_name = csv_path.stem

    # Adding the new column 'plate_name' to the left of df_merged
    df_merged.insert(0, "plate_name", plate_name)

    return df_merged


def save_summary_stats(results_directory):
    """Extracts summary stats from per object results and concatenates all df into a final_df (+save to disk as .csv) containing all plates data"""
    # Define the directory containing the copied per_organoid_stats
    csv_results_path = Path(os.path.join(results_directory, "per_organoid_stats"))

    # Initialize an empty DataFrame to collect all summary dataframes
    final_df = pd.DataFrame()

    # Scan for .csv files in each subdirectory
    for csv_path in csv_results_path.glob("*.csv"):
        summary_df = extract_summary_stats(csv_path)
        # Append the summary_df to the final_df
        final_df = pd.concat([final_df, summary_df], ignore_index=True)

    # Create the summary_stats directory if it doesn't exist
    summary_stats_directory = Path(results_directory, "summary_stats")
    summary_stats_directory.mkdir(parents=True, exist_ok=True)

    # Save the final_df as a .csv file under the summary_stats directory
    final_csv_path = summary_stats_directory / "summary_stats.csv"
    final_df.to_csv(final_csv_path, index=False)


def find_focus(images_per_well):
    """Processes all the images and extract the number of organoids in focus from each image"""

    nr_infocus_organoids = {}

    for well_id in tqdm(images_per_well):
        imgs_to_process = images_per_well[well_id]

        for input_img in imgs_to_process:
            # Check if the well_id exists in the dictionary, if not, create a new list
            if well_id not in nr_infocus_organoids:
                nr_infocus_organoids[well_id] = []

            # Load one RGB image and transform it into grayscale (if needed) for APOC
            rgb_img = load_image(input_img)
            if len(rgb_img.shape) < 3:
                img = rgb_img
            elif rgb_img.shape[2] == 3:
                img = rgb2gray(rgb_img)
            else:
                print(
                    "Modify the loader to accommodate different file formats",
                    rgb_img.shape,
                )

            # Apply object segmenter from APOC
            try:
                segmenter = ObjectSegmenter(opencl_filename="./ObjectSegmenter.cl")
                result = segmenter.predict(image=img)
            except IndexError:
                segmenter = ObjectSegmenter(
                    opencl_filename="./pretrained_APOC/ObjectSegmenter.cl"
                )
                result = segmenter.predict(image=img)

            # Closing some holes in the organoid labels
            closed_labels = cle.closing_labels(result, None, radius=4.0)

            # Exclude small labels, cutout in pixel area seems to be below 1000px
            exclude_small = cle.exclude_small_labels(closed_labels, None, 1000.0)
            exclude_small = np.array(
                exclude_small, dtype=np.int32
            )  # Change dtype of closed labels to feed array into nsbatwm.split

            # Splitting organoids into a binary mask
            split_organoids = nsbatwm.split_touching_objects(exclude_small, sigma=10.0)

            # Connected component (cc) labeling
            cc_split_organoids = nsbatwm.connected_component_labeling(
                split_organoids, False
            )

            # Apply object classifier from APOC
            try:
                classifier = ObjectClassifier(opencl_filename="./ObjectClassifier.cl")
                result = classifier.predict(labels=cc_split_organoids, image=img)
            except AttributeError:
                classifier = ObjectClassifier(
                    opencl_filename="./pretrained_APOC/ObjectClassifier.cl"
                )
                result = classifier.predict(labels=cc_split_organoids, image=img)

            # Convert the resulting .cle image into a np.array to count objects within each class
            image_array = np.array(result, dtype=np.int8)

            # Create masks for each class
            background_mask = image_array == 0
            out_of_focus_mask = image_array == 1
            in_focus_mask = image_array == 2

            # Label connected components in each mask
            background_labels = measure.label(background_mask, connectivity=2)
            out_of_focus_labels = measure.label(out_of_focus_mask, connectivity=2)
            in_focus_labels = measure.label(in_focus_mask, connectivity=2)

            # Calculate the number of objects in each class
            num_background_objects = np.max(background_labels)
            num_out_of_focus_objects = np.max(out_of_focus_labels)
            num_in_focus_objects = np.max(in_focus_labels)

            # print(f"Number of Background Objects: {num_background_objects}")
            # print(f"Number of Out-of-Focus Objects: {num_out_of_focus_objects}")
            # print(f"Number of In-Focus Objects: {num_in_focus_objects}")
            try:
                in_focus_percentage = (
                    num_in_focus_objects
                    / (num_in_focus_objects + num_out_of_focus_objects)
                ) * 100
                # print(f"Percentage of In-Focus Objects: {in_focus_percentage}")
            except ZeroDivisionError:
                in_focus_percentage = 0

            nr_infocus_organoids[well_id].append(in_focus_percentage)

    return nr_infocus_organoids


def find_highest_infocus(nr_infocus_organoids):
    """Finds the z-stack with the most organoids in-focus in the dictionary containing said data"""
    max_index_dict = (
        {}
    )  # Create a new dictionary to store the maximum index for each key

    for key, values in nr_infocus_organoids.items():
        max_value = max(values)  # Find the maximum value within the list
        max_index = values.index(max_value)  # Get the index of the maximum value
        max_index_dict[key] = max_index  # Store the index in the new dictionary

    return max_index_dict


def store_imgs(
    images_per_well, max_index_dict, output_dir="./output/in_focus_organoids"
):
    """Stores images in focus"""
    # Create a directory to store the tif files if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through keys in max_index_dict
    for key, index in max_index_dict.items():
        if key in images_per_well:
            file_paths = images_per_well[key]
            if 0 <= index < len(file_paths):
                file_path = file_paths[index]

                # Construct the output file path
                output_path = os.path.join(output_dir, f"{key}.tif")

                tifffile.imwrite(output_path, load_image(file_path))

                # print(f"Saved {key}.tif to {output_path}")
            # else:
            # print(f"Invalid index for key {key}: {index}")
        # else:
        # print(f"Key {key} not found in images_per_well dictionary")


def plot_plate(
    resolution, output_path, img_folder_path, show_fig=True, colormap="gray"
):
    """Plot images in a grid-like fashion according to the well_id position in the plate"""
    import matplotlib.pyplot as plt

    img_folder_path = Path(img_folder_path)
    output_path = Path(output_path)

    # Initialize a dictionary to store images by rows (letters)
    image_dict = {}

    # Iterate through the image files in the folder
    for image_path in sorted(img_folder_path.glob("*.tif")):
        filename = image_path.name
        if filename.endswith(".tif"):
            # Extract the first letter and the number from the filename
            first_letter = filename[0]
            number = int(filename[1:3])

            # Create a dictionary entry for the first letter if it doesn't exist
            if first_letter not in image_dict:
                image_dict[first_letter] = {}

            # Create a dictionary entry for the number if it doesn't exist
            if number not in image_dict[first_letter]:
                image_dict[first_letter][number] = []

            # Append the image filename to the corresponding number
            image_dict[first_letter][number].append(filename)

    # Sort the dictionary by keys (letters) and nested dictionary by keys (numbers)
    sorted_image_dict = {
        letter: dict(sorted(images.items()))
        for letter, images in sorted(image_dict.items())
    }

    # Calculate the number of rows based on the number of letters
    num_rows = len(sorted_image_dict)

    if not sorted_image_dict:
        raise ValueError(f"No .tif well images found in: {img_folder_path}")

    # Calculate the number of columns based on the maximum number
    num_cols = max(max(images.keys()) for images in sorted_image_dict.values())

    # Calculate the figsize based on the number of columns and rows
    fig_width = num_cols * 3  # Adjust the multiplier as needed
    fig_height = num_rows * 2.5  # Adjust the multiplier as needed

    # Create a subplot for each image, using None for empty subplots
    if num_rows == 1:
        fig, axes = plt.subplots(
            1, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
    elif num_cols == 1:
        fig, axes = plt.subplots(
            num_rows, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
    else:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(fig_width, fig_height),
            sharex=True,
            sharey=True,
        )

    for i, (letter, images) in enumerate(sorted_image_dict.items()):
        for j, (number, filenames) in enumerate(images.items()):
            if filenames:
                image_filename = filenames[0]  # Use the first filename in the list
                image = load_image(img_folder_path / image_filename)
                show_kwargs = (
                    {} if image.ndim == 3 and image.shape[2] >= 3 else {"cmap": colormap}
                )

                if num_rows == 1:
                    axes[j].imshow(image, **show_kwargs)
                    axes[j].set_title(f"{letter}{number:02d}")
                    axes[j].axis("off")
                elif num_cols == 1:
                    axes[i].imshow(image, **show_kwargs)
                    axes[i].set_title(f"{letter}{number:02d}")
                    axes[i].axis("off")
                else:
                    axes[i, j].imshow(image, **show_kwargs)
                    axes[i, j].set_title(f"{letter}{number:02d}")
                    axes[i, j].axis("off")

            else:
                # If there are no images for a specific letter-number combination, remove the empty subplot
                if num_rows == 1:
                    fig.delaxes(axes[j])
                elif num_cols == 1:
                    fig.delaxes(axes[i])
                else:
                    fig.delaxes(axes[i, j])

    # Adjust the spacing and set aspect ratio to be equal
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save the plot at a higher resolution
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="tif", dpi=resolution, bbox_inches="tight")
    plt.close(fig)

    # If False plt.show() is not run to avoid loop stop upon display
    if show_fig:
        # Show the plot (optional)
        plt.show()


def segment_organoids(in_focus_organoids):
    """Processes individual z-stacks inside a folder and returns a dictionary containing organoid object masks stored as StackViewNDArray arrays"""

    segmented_organoids = {}
    images = sorted(in_focus_organoids.glob("*.tif"))

    # Iterate through the files in the directory
    for input_img in tqdm(images):
        # Get the filename without the extension
        filename = input_img.stem

        # Load one RGB image and transform it into grayscale (if needed) for APOC
        rgb_img = load_image(input_img)
        if len(rgb_img.shape) < 3:
            img = rgb_img
        elif rgb_img.shape[2] == 3:
            img = rgb2gray(rgb_img)
        else:
            print(
                "Modify the loader to accommodate different file formats",
                rgb_img.shape,
            )

        # Apply object segmenter from APOC
        try:
            segmenter = ObjectSegmenter(opencl_filename="./ObjectSegmenter.cl")
            result = segmenter.predict(image=img)
        except IndexError:
            segmenter = ObjectSegmenter(
                opencl_filename="./pretrained_APOC/ObjectSegmenter.cl"
            )
            result = segmenter.predict(image=img)

        # Closing some holes in the organoid labels
        closed_labels = cle.closing_labels(result, None, radius=4.0)

        # Exclude small labels, cutout in pixel area seems to be below 1000px
        exclude_small = cle.exclude_small_labels(closed_labels, None, 1000.0)
        exclude_small = np.array(
            exclude_small, dtype=np.int32
        )  # Change dtype of closed labels to feed array into nsbatwm.split

        # Splitting organoids into a binary mask
        split_organoids = nsbatwm.split_touching_objects(exclude_small, sigma=10.0)

        # Connected component (cc) labeling
        cc_split_organoids = nsbatwm.connected_component_labeling(
            split_organoids, False
        )

        # Remove labels on edges
        edge_removed = nsbatwm.remove_labels_on_edges(cc_split_organoids)

        segmented_organoids[filename] = edge_removed

    return segmented_organoids


def segment_in_focus_organoids(in_focus_organoids):
    """Processes individual z-stacks inside a folder and returns a dictionary containing in-focus/out-of-focus organoid object masks stored as StackViewNDArray arrays"""

    focus_masks = {}
    images = sorted(in_focus_organoids.glob("*.tif"))

    # Iterate through the files in the directory
    for input_img in tqdm(images):
        # Get the filename without the extension
        filename = input_img.stem

        # Load one RGB image and transform it into grayscale (if needed) for APOC
        rgb_img = load_image(input_img)
        if len(rgb_img.shape) < 3:
            img = rgb_img
        elif rgb_img.shape[2] == 3:
            img = rgb2gray(rgb_img)
        else:
            print(
                "Modify the loader to accommodate different file formats",
                rgb_img.shape,
            )

        # Apply object segmenter from APOC
        try:
            segmenter = ObjectSegmenter(opencl_filename="./ObjectSegmenter.cl")
            result = segmenter.predict(image=img)
        except IndexError:
            segmenter = ObjectSegmenter(
                opencl_filename="./pretrained_APOC/ObjectSegmenter.cl"
            )
            result = segmenter.predict(image=img)

        # Closing some holes in the organoid labels
        closed_labels = cle.closing_labels(result, None, radius=4.0)

        # Exclude small labels, cutout in pixel area seems to be below 1000px
        exclude_small = cle.exclude_small_labels(closed_labels, None, 1000.0)
        exclude_small = np.array(
            exclude_small, dtype=np.int32
        )  # Change dtype of closed labels to feed array into nsbatwm.split

        # Splitting organoids into a binary mask
        split_organoids = nsbatwm.split_touching_objects(exclude_small, sigma=10.0)

        # Connected component (cc) labeling
        cc_split_organoids = nsbatwm.connected_component_labeling(
            split_organoids, False
        )

        # Remove labels on edges
        edge_removed = nsbatwm.remove_labels_on_edges(cc_split_organoids)

        # Apply object classifier from APOC
        try:
            classifier = ObjectClassifier(opencl_filename="./ObjectClassifier.cl")
            result = classifier.predict(labels=edge_removed, image=img)
        except AttributeError:
            classifier = ObjectClassifier(
                opencl_filename="./pretrained_APOC/ObjectClassifier.cl"
            )
            result = classifier.predict(labels=edge_removed, image=img)

        focus_masks[filename] = result

    return focus_masks


def save_object_mask(segmented_organoids, output_directory):
    # Iterate through the dictionary and save each array as a .tif file
    for filename, object_mask in segmented_organoids.items():
        # Create a filename for the .tif file within the output_directory
        save_filename = os.path.join(output_directory, f"{filename}.tif")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)

        # Save the cc_split_organoid array as a grayscale .tif file
        tifffile.imwrite(
            save_filename,
            object_mask,
            photometric="minisblack",
            dtype=object_mask.dtype,
        )


def random_cmap():
    from matplotlib.colors import ListedColormap

    np.random.seed(42)
    cmap = ListedColormap(np.random.rand(256, 4))
    # value 0 should just be transparent
    cmap.colors[:, 3] = 0.5
    cmap.colors[0, :] = 1
    cmap.colors[0, 3] = 0

    # if image is a mask, color (last value) should be red
    cmap.colors[-1, 0] = 1
    cmap.colors[-1, 1:3] = 0
    return cmap


# --------------- PARALLELIZATION FUNCTIONS ---------------- #


def process_folder(folder, parent_folder, username):
    """Reads a folder, scans all z-stacks per well and stores a copy of the in-focus stack"""
    directory_path = parent_folder.joinpath(folder)
    print(directory_path)

    # The following function will read all the images contained within the directory_path above
    # and store them grouped by well_id.
    images_per_well = read_images(directory_path)

    # Compute the nr of organoids in focus per well
    nr_infocus_organoids = find_focus(images_per_well)

    # Store a .csv copy of the max_index_dict containing the percentages of organoids in focus
    # Create a Pandas DataFrame
    df = pd.DataFrame(nr_infocus_organoids)

    # Specify the output directory path
    directory = Path(f"./output/{username}")
    output_directory = directory.joinpath(folder)

    # Check if the output directory already exists and create it if it does not
    try:
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{output_directory}' already exists.")

    # Save the DataFrame as a .csv file
    df.to_csv(
        f"./{str(output_directory)}/Percentage_in_focus_per_well_{str(folder)}.csv",
        index=False,
    )

    # Finding the z-stack with the most organoids in-focus
    max_index_dict = find_highest_infocus(nr_infocus_organoids)

    # In case one of the wells has no detectable organoids in focus, this will substitute the focal plane
    # with an average of all focal planes in the plate

    # Calculate the average of all values in the dictionary
    average_value = round(sum(max_index_dict.values()) / len(max_index_dict))

    # Substitute the 0 values for the average focal plane
    for well, in_focus_stack in max_index_dict.items():
        if in_focus_stack == 0:
            max_index_dict[well] = average_value

    # Storing a copy of each z-stack with the most organoids in focus
    store_imgs(
        images_per_well,
        max_index_dict,
        output_dir=f"{output_directory}/in_focus_organoids",
    )


def save_organoid_segmentation(in_focus_organoids):
    """Reads a folder containing grayscale images, segments the organoids and saves the resulting masks in a new folder"""
    # segment_organoids() returns a dictionary where the organoid labels are stored under each well_id key
    segmented_organoids = segment_organoids(Path(in_focus_organoids))

    # Define the directory path where you want to save the segmented organoid masks
    # Split the in_focus_organoids path to obtain the folder that is one level up (head)
    head, tail = os.path.split(in_focus_organoids)
    output_directory = os.path.join(head, "segmented_organoids")

    # Save the segmented organoid masks contained in segmented_organoids in the above defined output directory
    save_object_mask(segmented_organoids, output_directory)


def save_focus_segmentation(in_focus_organoids):
    """Reads a folder containing grayscale images, segments the organoids and saves the resulting masks in a new folder"""
    # segment_in_focus_organoids() returns a dictionary where the organoid labels are stored under each well_id key
    focus_classified_organoids = segment_in_focus_organoids(Path(in_focus_organoids))

    # Define the directory path where you want to save the segmented organoid masks
    # Split the in_focus_organoids path to obtain the folder that is one level up (head)
    head, tail = os.path.split(in_focus_organoids)
    output_directory = os.path.join(head, "in_out_focus_masks")

    # Save the segmented organoid masks contained in segmented_organoids in the above defined output directory
    save_object_mask(focus_classified_organoids, output_directory)
