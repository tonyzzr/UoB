# Multi-View Ultrasound Fusion and Tracking (Project Name TBD)

## Project Description

This project aims to develop a robust pipeline for processing multi-view, dual-frequency (high and low resolution) ultrasound data acquired from a specialized imaging device. The core objectives are:

1.  **Preprocessing:** Convert raw `.mat` ultrasound data into a structured dataset format suitable for deep learning analysis.
2.  **Feature Extraction & Matching:** Leverage deep foundation models and feature upsampling techniques (e.g., FeatUp) to extract and match robust semantic features across different views, time sequences, and frequencies (resolutions).
3.  **Image Registration & Fusion:** Utilize the identified feature correspondences to accurately register images from different views within the same time frame and fuse them into a comprehensive, holistic representation of the scanned tissue.
4.  **Temporal Tracking:** Implement pixel-level tracking of anatomical or architectural features within the fused, holistic view over time sequences.

The ultimate goal is to provide a tool for analyzing tissue structure and dynamics from complex, multi-modal ultrasound recordings. The codebase emphasizes modularity and flexibility to accommodate different datasets, feature extraction models, registration algorithms, and analysis pipelines.

*Note: Existing code from the initial implementation has been moved to `UoB/legacy` and should be used for reference purposes only. The goal is to rewrite and improve upon this code within the new structure.*

## Proposed Codebase Structure

```
.
├── configs/                  # Configuration files (TOML format) for pipelines, models, datasets
│   ├── dataset/
│   │   └── recording_xyz.toml
│   ├── features/
│   │   └── dino_featup.toml
│   ├── registration/
│   │   └── default_ransac.toml
│   └── pipelines/
│       └── full_process.toml
├── data/                     # Raw and processed data storage (or symlinks)
│   ├── raw/                  # Original .mat files (or organized by recording)
│   │   └── recording_2022-08-17_trial2-arm/
│   │       ├── 1_HF.mat
│   │       └── 1_LF.mat
│   └── processed/            # Processed datasets & Intermediate Files (structured)
│       └── recording_2022-08-17_trial2-arm/
│           ├── combined_mvbv.pkl                 # Base processed data
│           ├── features/                  # Example intermediate feature files
│           │   └── 1_lftx_frame0_dino_featup.pt
│           ├── matches/                   # Example intermediate matches
│           │   └── 1_lftx0_hftx0_matches.npy
│           ├── poses/                     # Example intermediate poses
│           │   └── 1_lftx_hftx_pose_ransac_v1.toml
│           └── visualizations/            # Example output visualizations
│               └── 1_fused_sequence.mp4
├── scripts/                  # Standalone scripts for specific tasks
│   ├── preprocess_data.py    # Script to run preprocessing pipeline
│   ├── extract_features.py   # Script for feature extraction
│   ├── run_registration.py   # Script for registration/fusion
│   ├── run_tracking.py       # Script for tracking
│   └── visualize.py          # Script for various visualizations
├── src/                      # Main source code
│   ├── UoB/                  # Core library modules (can be renamed)
│   │   ├── __init__.py
│   │   ├── data/             # Data loading, representation (e.g., MultiViewBmodeVideo)
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py     # Dataset classes (potentially using registry)
│   │   │   ├── formats.py      # Data structure definitions (e.g., dataclasses)
│   │   │   └── readers.py      # Readers for specific raw formats (e.g., .mat)
│   │   ├── pipelines/        # End-to-end processing pipelines (config-driven)
│   │   │   ├── __init__.py
│   │   │   ├── base_pipeline.py
│   │   │   └── registration_fusion.py
│   │   ├── preprocessing/    # Preprocessing steps (mat -> bmode -> mvbv)
│   │   │   ├── __init__.py
│   │   │   └── mat_converter.py
│   │   ├── features/         # Feature extraction modules
│   │   │   ├── __init__.py
│   │   │   ├── extractors.py   # Feature extractor classes (registry?)
│   │   │   ├── upsamplers.py   # Upsampling modules (e.g., FeatUp wrapper)
│   │   │   └── matching.py     # Correspondence computation
│   │   ├── registration/     # Registration algorithms
│   │   │   ├── __init__.py
│   │   │   ├── estimators.py   # Pose estimation (registry?)
│   │   │   └── fusion.py       # Image fusion techniques
│   │   ├── tracking/         # Temporal tracking algorithms
│   │   │   └── __init__.py
│   │   ├── visualization/    # Plotting and visualization utilities
│   │   │   ├── __init__.py
│   │   │   ├── plot_features.py
│   │   │   └── plot_correspondence.py
│   │   └── utils/            # Common utilities (geometry, math, io)
│   │       ├── __init__.py
│   │       ├── geometry.py
│   │       └── io.py
│   └── registries.py         # Central registries for datasets, models, etc. (optional)
├── notebooks/                # Jupyter notebooks for experimentation and analysis
│   ├── 1_preprocessing.ipynb
│   ├── 2_feature_exploration.ipynb
│   ├── 3_registration_fusion.ipynb
│   └── 4_tracking.ipynb
├── webapp/                   # Web application for interactive visualization
│   ├── backend/              # FastAPI backend 
│   │   ├── app/
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend/             # React/Vue/Svelte frontend
│       ├── public/
│       ├── src/
│       └── package.json
├── third_party/              # External libraries/code (e.g., FeatUp, part_cosegmentation)
├── tests/                    # Unit and integration tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file

```

**Key Design Principles:**

*   **Configuration Driven:** Use TOML files in `configs/` to define dataset paths, model parameters, and pipeline steps. This avoids hardcoding values in scripts.
*   **Modularity:** Each component (data loading, preprocessing, feature extraction, registration, tracking, visualization) resides in its own directory within `src/UoB/`.
*   **Intermediate File Management:** Processed data and intermediate results (features, matches, poses, visualizations) are stored systematically within `data/processed/<recording_id>/<stage>/`. Filenames should be descriptive (e.g., `<id>_<view/freq>_<frame>_<model/step>.<ext>`). Pipelines should leverage caching by checking for existing intermediate files based on configuration and input data timestamps/hashes.
    *   *Formats:* Use appropriate formats (e.g., `.pt` for features, `.npy` for matches, `.toml` for poses, `.pkl` for complex objects, `.png`/`.mp4` for visualizations).
*   **Registry Pattern (Recommended):** Implement simple registries (e.g., dictionaries mapping names to classes/functions) in `src/registries.py` or within specific modules (`features/extractors.py`, `registration/estimators.py`). This allows selecting models or algorithms via configuration files (e.g., `feature_extractor = "dino16_featup"` in a TOML file).
*   **Clear Data Flow:** Define clear data structures (e.g., using `dataclasses` in `src/UoB/data/formats.py`) that are passed between pipeline stages.
*   **Separation of Concerns:** Scripts in `scripts/` orchestrate pipelines defined in `src/UoB/pipelines/`, using components from other `src/` modules. Notebooks are for exploration, referencing the `src/` code. The `webapp/` directory contains the separate interactive visualization tool.

## Development Plan (Core Library + Integrated Web Application)

This plan outlines the development of the core processing library (`src/UoB/`) and the parallel development of an interactive web application (`webapp/`) for visualization.

**Phase 1: Refactoring and Setup**

1.    [x] **Establish Directory Structure:** Create the proposed directory structure, including `src/`, `scripts/`, `configs/`, `data/`, `tests/`, `notebooks/`, and `webapp/`.
2.    [ ] **Move & Refactor Existing Code:** Relocate relevant logic from `UoB/legacy` to appropriate locations within the new `src/UoB/` structure, rewriting and improving as needed. *Do not simply copy-paste legacy code.*
    *   `mat_to_mvbv_converter.py` logic -> `src/UoB/preprocessing/mat_converter.py`
    *   `visualize_*.py` logic -> `src/UoB/visualization/` (refactor into reusable functions)
    *   `data/*.py` logic -> `src/UoB/data/` (refactor `bmode.py`, `multiview_bmode.py`, define formats in `formats.py`)
    *   `model/*.py` logic -> `src/UoB/registration/` and `src/UoB/utils/`
    *   `tissue_structure_coseg.py` logic -> Potentially `src/UoB/features/segmentation.py` or refactor if needed.
3.    [ ] **Refactor Imports:** Update all import statements to reflect the new structure.
4.    [ ] **Setup `requirements.txt`:** Consolidate all dependencies.
5.    [x] **Implement Basic Configuration:** Set up TOML loading (`tomllib` or `toml`) for dataset paths and basic parameters.
6.    [x] **Refactor Preprocessing:**
    *   Create a `MatConverter` class in `src/UoB/preprocessing/mat_converter.py`.
    *   Make it configurable (input dir, output dir, config paths) via args or a config file.
    *   Create `scripts/preprocess_data.py` to run the conversion, incorporating intermediate file checking/saving.
7.    [x] **Refactor Data Loading:**
    *   Define `MultiViewBmodeVideo` (and related structures) formally using `dataclasses` in `src/UoB/data/formats.py`.
    *   Create a generic `UltrasoundDataset` class in `src/UoB/data/datasets.py` capable of loading the processed `.pkl` (or chosen format) files based on a config.
8.  **[WEB APP]** [x] **Setup Basic Web App Structure:** Initialize backend (e.g., FastAPI in `webapp/backend`) and frontend (e.g., React in `webapp/frontend`) projects, setup basic API communication (CORS, placeholder endpoints), create UI shell.
9.  **[WEB APP]** [x] **Build Dataset Explorer:** Create UI component in the frontend to list available processed recordings (`data/processed/*`). Implement a backend API endpoint to serve recording metadata (list directories/files).

**Phase 2: Feature Extraction and Visualization**

10. [x] **Integrate Feature Extractors:**
    *   [x] Implement foundation model wrapper classes in `src/UoB/features/extractors.py` and `src/UoB/features/upsamplers.py`.
    *   [x] Make model selection configurable (e.g., specify `dino16` via config).
    *   [x] Implement registry pattern for model selection.
11. [x] **Refactor Visualization Code:**
    *   [x] Move PCA logic and plotting functions from `visualize_features.py` to `src/UoB/visualization/plot_features.py`.
    *   [x] Move correspondence computation and plotting from `visualize_similarity.py` to `src/UoB/features/matching.py` and `src/UoB/visualization/plot_correspondence.py`.
    *   [x] Make visualization functions accept data objects (e.g., `MultiViewBmodeVideo` frames) and feature tensors.
12. [x] **Create Feature Extraction Script:** 
    *   [x] Developed `scripts/extract_features.py` that loads data, selects a feature extractor via config, computes features, and saves them to disk.
    *   [x] Implemented command-line arguments for configuration and frame selection.
    *   [x] Added capability to process frames in batches with status tracking.
13. [x] **Create Visualization Script:** Develop `scripts/visualize.py` with subcommands (e.g., `visualize.py features`, `visualize.py correspondence`) driven by configs.
14. [ ] **Address Memory Issues:** Systematically apply memory management techniques (CPU/GPU transfer, `del`, `gc.collect`, `torch.cuda.empty_cache`) where needed.
15. **[WEB APP]** [x] **Visualize Raw Frames:** Enhance Dataset Explorer UI to select a recording and view individual frames (LF/HF, different views) from its `MultiViewBmodeVideo` file.
16. **[WEB APP]** [x] **Visualize Extracted Features:** 
    *   [x] Implemented on-demand feature extraction in the data service to eliminate the need to store thousands of large feature files.
    *   [x] Created API endpoint to visualize features with PCA computation.
    *   [x] Developed UI component to display 4x8 grid of original and PCA-visualized features.
17. **[WEB APP]** [x] **Visualize Correspondences:** 
    *   [x] Implemented interactive correspondence visualization UI with source view/point selection.
    *   [x] Added backend API endpoints for computing correspondences and generating visualizations.
    *   [x] Created Next.js API routes to forward requests to Python backend.
    *   [x] Implemented matplotlib-based grid visualization in the backend to eliminate client-side coordinate system issues.
    *   [x] Added loading states and error handling for a robust user experience.

**Phase 3: Registration and Fusion**

18. [ ] **Refactor Registration Code:**
    *   Move pose estimation logic (`rela_pose_est.py`, potentially parts of notebooks) into `src/UoB/registration/estimators.py`. Aim for classes that take features/matches as input and return poses.
    *   Consider a registry for different estimation methods.
19. [ ] **Implement Fusion:** Move/refactor image fusion logic (`image_fusion.py`) into `src/UoB/registration/fusion.py`.
20. [ ] **Create Registration Pipeline:** Develop `scripts/run_registration.py` or a pipeline class in `src/UoB/pipelines/` that:
    *   Loads data and features.
    *   Performs matching (using `src/UoB/features/matching.py`).
    *   Estimates poses (using `src/UoB/registration/estimators.py`).
    *   Applies poses and fuses images (using `src/UoB/registration/fusion.py`).
    *   Saves results (registered point clouds, fused images) following the intermediate file strategy.
21. **[WEB APP]** [ ] **Visualize Registration Results:** Add UI components to display fused images/point clouds. Allow overlaying original views based on calculated poses. Requires backend API endpoints to serve fused data and poses.

**Phase 4: Tracking**

22. [ ] **Research/Select Tracking Algorithm:** Choose a suitable pixel/feature tracking algorithm (e.g., optical flow, particle filters, deep tracking methods).
23. [ ] **Implement Tracking Module:** Create `src/UoB/tracking/` module with the chosen algorithm. It should operate on the fused image sequence.
24. [ ] **Create Tracking Script/Pipeline:** Develop `scripts/run_tracking.py` to apply tracking to a fused video sequence and save results.
25. **[WEB APP]** [ ] **Visualize Tracking Results:** Add UI component to display fused video sequences with tracked points/features overlaid. Requires backend API endpoint to serve video data and tracking coordinates.

**Phase 5: Polish and Testing**

26. [ ] **Add Unit Tests:** Implement tests in `tests/` for key components (data loading, feature extraction, registration steps).
27. [ ] **Documentation:** Add docstrings to classes and functions. Refine `README.md`.
28. [ ] **Refine Configuration System:** Enhance the configuration system (e.g., using Hydra or a similar library) for more complex experiments.
29. [ ] **Cleanup Notebooks:** Update notebooks in `notebooks/` to use the refactored `src/` code, serving as examples and analysis tools.
29. **[WEB APP]** [ ] **Refine Web App UI/UX:** Improve navigation, add loading indicators, error handling, polish visualizations, and potentially add interactive parameter adjustment controls.