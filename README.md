# intestinal_organoid_yolov8
Tools to analyze intestinal organoid imaging at the Martin-Alonso/Oudhoff lab using AI

<h2>Instructions</h2>

1. In order to run these Jupyter notebooks and .py scripts you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a virtual environment (venv) either using the following command or recreate the environment from the .yml file you can find in the envs folder:

    <code>mamba create -n int_organoids python=3.9 devbio-napari cellpose pytorch torchvision plotly pyqt ultralytics -c conda-forge -c pytorch</code>

3. To recreate the venv from the environment.yml file stored in the envs folder (recommended) navigate into the envs folder using <code>cd</code> in your console and then execute:

    <code>mamba env create -f environment.yml</code>

4. Once your virtual environment is ready you can copy all your folders containing the images from EVOS inside the data directory using the following structure:

    <code>
    intestinal_organoid_YOLOv8/   #Primary data folder for the project
    ├── data/                     #All input data is stored here. 
    │   ├── Plate_01/
    │   │   ├── P1_Plate_M_p00_z00_0_A01f00d0.TIF
    │   │   ├── P1_Plate_M_p00_z01_0_A01f00d0.TIF
    │   │   └── ...
    │   ├── Plate_02/
    │   │   ├── P2_Plate_M_p00_z00_0_A01f00d0.TIF
    │   │   ├── P2_Plate_M_p00_z01_0_A01f00d0.TIF
    │   │   └── ...
    │   └── ...
    </code>

5. The easiest way to interact with the analysis code is via Jupyter Lab. To launch a jupyter lab server run the following commands:
    <code>mamba activate int_organoids</code>
    <code>jupyter_lab</code>

6. Open 1_image_analysis.ipynb, define your username and desired resolution for the output plates and run all the cells.