# Changelog

## 2025-04-17

### Phase 2 & 3: Feature Matching & Correspondence

- [ ] **Refactor Matching Logic (from Task 11):**
    - [ ] Migrate correspondence computation logic from `legacy/visualize_similarity.py` to `src/UoB/features/matching.py`. Define methods for computing similarity/correlation between extracted deep features.
- [ ] **Refactor Correspondence Visualization (from Task 11):**
    - [ ] Migrate correspondence plotting logic from `legacy/visualize_similarity.py` to `src/UoB/visualization/plot_correspondence.py`. Ensure functions accept feature tensors and match data.
- [ ] **Develop Correspondence Visualization Script (Task 13):**
    - [ ] Start implementing the `visualize.py correspondence` subcommand.
    - [ ] It should load features (potentially using `extract_features.py` logic or saved files), compute matches (using `src/UoB/features/matching.py`), and visualize them (using `src/UoB/visualization/plot_correspondence.py`).
- [ ] **Plan/Start Web App Correspondence Backend (Task 17):**
    - [ ] Design the API endpoint required for visualizing correspondences interactively in the web app.
    - [ ] Define how the backend will compute or load correspondence data on demand.

## 2025-04-16

### Web App Enhancements & Optimization

- [x] **Improve UI Appearance:** Refactor the main page layout for better visual appeal.
- [x] **Implement Grid View:** Display all 16 spatial views (8 LF + 8 HF) in a 2x8 grid for a selected frame.
- [x] **Add Frame Slider:** Implement a slider component to allow users to easily select the frame index to visualize.
- [x] **Redesign Color Scheme:** Update the UI to match the "Item Organizer" style with:
  - Light gray background with clean white content areas
  - Professional blue accent colors for interactive elements
  - Consistent rounded corners and subtle shadows
  - More minimal, modern aesthetic overall
- [x] **Enhance Layout:** Improve spacing, typography and visual hierarchy
- [x] **Add Navigation Header:** Include a proper navigation bar at the top for better UX
- [x] **Temporarily Hide Date:** Modified UI display logic to hide dates in recording names.
- [ ] **Responsiveness:** Ensure layout works well on different screen sizes (Deferred for now)
- [x] **Implement PKL Caching:** Modify the data service to load the PKL file only once and keep it in memory
  - Added in-memory LRU cache with time-based expiration
  - Added cache status endpoint and manual clear function 
  - Significantly reduces load time for subsequent frame requests
- [ ] **Batch Image Loading:** Investigate pre-loading neighboring frames to improve perceived performance
- [ ] **Add Loading States:** Enhance UI feedback during image loading operations

### Phase 2: Feature Extraction - Recent Progress

- [x] **Setup Feature Extraction Structure:**
    *   [x] Create base classes (`BaseFeatureExtractor`, `BaseFeatureUpsampler`) in `src/UoB/features/`.
    *   [x] Implement registry system (`src/registries.py`) and integrate with feature modules.
    *   [x] Implement placeholder classes (`DinoV1Extractor`, `DinoV1JBUpsampler`) registered with appropriate keys.
    *   [x] Implement custom padding transform (`PadToSquareAndAlign`) in `src/UoB/utils/transforms.py`.
    *   [x] Implement shared preprocessing pipeline (padding, normalization) in base/placeholder classes.
    *   [x] Create initial `pytest` structure in `tests/features/` and migrate initial tests.
- [ ] **Integrate Feature Extractors:** (Task 10 from overall plan - Backbone part deferred)
    *   ~Wrap foundation models (DINOv2?) into classes in `src/UoB/features/extractors.py`.~ (Placeholder exists)
    *   ~Use a **registry pattern**.~ (Done)
    *   ~Make model selection configurable (e.g., specify `dino16` via **TOML config**).~ (Planned)
    *   [ ] **(Deferred)** Implement actual backbone model loading (`_load_model`) and forward pass (`forward`) in `DinoV1Extractor` / other backbone-only extractors. (User has alternative solution).
    *   ~**Standardize Preprocessing:**~ (Done in placeholder)
    *   ~**Configure Model Loading:**~ (Deferred for backbone)
    *   ~**Integrate Data Formats:**~ (Placeholder accepts tensors, full integration later)
- [ ] **Integrate Feature Upsamplers:** (Related to Task 10)
    *   Wrap upsamplers (FeatUp) into classes in `src/UoB/features/upsamplers.py`. (Placeholder `JointBilateralUpsampler` exists)
    *   Use a **registry pattern**. (Done)
    *   Make upsampler selection configurable (e.g., specify `featup_jbu` with `backbone_hub_id = 'dino16'` via **TOML config**). (Planned)
    *   [x] **Implement JBU Loading:** Implement `JointBilateralUpsampler._load_model` using `torch.hub.load("mhamilton723/FeatUp", backbone_hub_id, ...)` based on registry key `'featup_jbu'` and configured `backbone_hub_id`.
    *   [x] **Implement JBU Forward:** Implement `JointBilateralUpsampler.forward` to correctly call the loaded FeatUp model.
    *   [x] **Refine Feature Dims:** Update `upsampled_feature_dim` property based on the actually loaded backbone (using attribute checks + fallback).
    *   [x] **Refine Preprocessing:** Use FeatUp's `norm` utility (`third_party.FeatUp.featup.util.norm`) for input normalization in `get_preprocessing_transform`.
    *   **Standardize Preprocessing:** (Done in placeholder)
    *   **Configure Model Loading:** (Done: hub call + backbone config via params)
    *   **Integrate Data Formats:** (Placeholder accepts tensors, full integration later)

- [x] **Refactor Visualization Code (related to features):** (Task 11 from overall plan)
    *   [x] **Move PCA & Plotting:** Migrate PCA calculation and feature plotting logic from `legacy/visualize_features.py` to `src/UoB/visualization/plot_features.py`. (Functions `apply_pca_to_features`, `fit_joint_pca`, `plot_feature_pca_comparison` created).
        *   [x] **Refine Joint PCA Logic:** Modify `plot_feature_pca_comparison` so that when `use_joint_pca=True`, it collects features, fits PCA once, and transforms all views with the same model.
        *   [x] **Fix PCA Dimension Bug:** Corrected transpose logic in `apply_pca_to_features` and `fit_joint_pca` to ensure PCA reduces the *feature* dimension (C) instead of spatial dimensions.
    *   [x] **Decouple Plotting:** Refactor the functions in `plot_features.py` to accept pre-computed feature tensors and relevant metadata. (Plotting function now takes features list).
    *   [x] **Web App Integration Link:** Ensure visualization logic supports or interfaces with requirements for Task 16 ([`[WEB APP] Visualize Extracted Features`](#phase-2-feature-extraction-and-visualization-overall-plan---reference)).
- [x] **Create Feature Extraction Script:** (Task 12 from overall plan)
    *   [x] Develop `scripts/extract_features.py` to:
        *   Load data using data loading functions.
        *   Select and configure feature extractor/upsampler via **TOML config** (leveraging the registry).
        *   Run feature extraction for all views in a frame range.
        *   Save extracted features systematically to `data/processed/<recording>/features/`.
- [x] **Web App Feature Visualization Enhancements:**
    *   [x] **Implement On-Demand Feature Extraction:** Modified data service to compute features on-the-fly instead of loading from files:
        *   Loads feature extractor model at server startup
        *   Processes frames on-demand when requested
        *   Eliminated the need to store thousands of large feature files
        *   Added performance tracking for extraction and PCA operations
    *   [x] **Fix Visualization UI:** Resolved issues with feature visualization endpoint and display.
- [ ] **Update Visualization Script:** (Task 13 from overall plan)
    *   [ ] Enhance `scripts/visualize.py` with a subcommand (e.g., `visualize.py features`) that:
        *   Loads pre-computed features for all 16 views of a frame.
        *   Uses the refactored functions in `src/UoB/visualization/plot_features.py` (including the 16-view joint PCA) for display.

### Phase 2: Feature Extraction and Visualization (Overall Plan - Reference)

10. [ ] **Integrate Feature Extractors:**
    *   Wrap foundation models (DINOv2?) and FeatUp into classes in `src/UoB/features/extractors.py` and `src/UoB/features/upsamplers.py`.
    *   Make model selection configurable (e.g., specify `dino16` via config).
11. [ ] **Refactor Visualization Code:**
    *   Move PCA logic and plotting functions from `visualize_features.py` to `src/UoB/visualization/plot_features.py`.
    *   Move correspondence computation and plotting from `visualize_similarity.py` to `src/UoB/features/matching.py` and `src/UoB/visualization/plot_correspondence.py`.
    *   Make visualization functions accept data objects (e.g., `MultiViewBmodeVideo` frames) and feature tensors.
12. [x] **Create Feature Extraction Script:** Develop `scripts/extract_features.py` that loads data, selects a feature extractor via config, computes features, and optionally saves them.
13. [ ] **Create Visualization Script:** Develop `scripts/visualize.py` with subcommands (e.g., `visualize.py features`, `visualize.py correspondence`) driven by configs.
14. [ ] **Address Memory Issues:** Systematically apply memory management techniques (CPU/GPU transfer, `del`, `gc.collect`, `torch.cuda.empty_cache`) where needed.
15. **[WEB APP]** [x] **Visualize Raw Frames:** Enhance Dataset Explorer UI to select a recording and view individual frames (LF/HF, different views) from its `MultiViewBmodeVideo` file. Requires backend API endpoint to load/serve specific frames.
    - **Setup Python Data Service:** Created a separate FastAPI service (`webapp/data_service`) to handle `.pkl` loading.
    - **Service Endpoints:** Implemented `/ping`, `/details`, and `/frames` endpoints in the Python service.
    - **Data Loading & Processing:** Added logic to load `combined_mvbv.pkl`, extract metadata, access specific frame tensors, normalize, and convert frames to PNG using Pillow.
    - **Debugging:** Resolved issues with TOML `null` parsing, `sys.path` configuration for `src.UoB` imports, log compression saturation (`max_value`), and data orientation (transpose) during preprocessing.
    - **Next.js API:** Modified details API route and created a new frame API route to proxy requests to the Python service.
    - **Frontend UI:** Added dropdowns/input for frame selection (frequency, spatial view, frame index) in `DatasetExplorer.tsx`.
    - **Frontend Display:** Implemented image display using an `<img>` tag pointing to the Next.js frame API route.
16. **[WEB APP]** [x] **Visualize Extracted Features:** Add UI functionality to display pre-computed feature visualizations (e.g., PCA plots) associated with a frame. Requires backend API endpoint to serve feature maps or their visualizations (e.g., load `.png` or compute/cache PCA on demand).
    - **Backend API:** Implemented `/recordings/{recording_id}/visualize_features/{frame_index}` endpoint with on-demand feature extraction.
    - **Memory Optimization:** Eliminated file-based storage by computing features at request time.
    - **Frontend UI:** Created `FeatureVisualizer.tsx` component for feature visualization display.
    - **Layout:** Implemented a 4x8 grid layout showing both original and PCA images for all 16 views.

## 2025-04-15

### Phase 1: Refactoring and Setup

- [x] **Establish Directory Structure:** Create the proposed directory structure, including `src/`, `scripts/`, `configs/`, `data/`, `tests/`, `notebooks/`, and `webapp/`.
- [ ] **Move & Refactor Existing Code:** Relocate relevant logic from `UoB/legacy` to appropriate locations within the new `src/UoB/` structure, rewriting and improving as needed. *Do not simply copy-paste legacy code.*
    - [ ] **Refactor Preprocessing:** Migrate `legacy/mat_to_mvbv_converter.py` logic to `src/UoB/preprocessing/mat_converter.py`.
        - [x] **Consolidate Processing:** Move image processing functions from `legacy/data/process.py` to `src/UoB/utils/processing.py`. (Added unit tests)
        - [x] **Consolidate Formats:** Move settings dataclasses (`MaskSetting`, etc.) and core data structures (`MatData`, `TransPos`, `Bmode`, `MultiViewBmodeVideo`) from `legacy/data/*` to `src/UoB/data/formats.py`. (Added unit tests)
        - [x] **Refactor Config:** Define TOML structure (`configs/preprocessing/default.toml`), implement TOML loading function (`utils/io.py`), refine `BmodeConfig.from_dict`. (Added unit tests for TOML loading)
        - [x] **Refactor Readers:** Moved `MatDataLoader` to `src/UoB/data/readers.py` and implemented `RecordingLoader` to load/concatenate full recordings. (Added unit tests)
        - [x] **Implement Converter:** Created `MatConverter` in `src/UoB/preprocessing/mat_converter.py` orchestrating loading, processing (incl. helper methods), and saving. (Added integration test)
        - [x] **Handle MVBV:** Integrated the conversion logic from `legacy/data/multiview_bmode.py::Bmode2MultiViewBmodeVideo` into the `MatConverter` class. // Completed as part of 'Implement Converter'.
    - [ ] **Refactor Data Structures & Loading:**
        - [ ] Define core data structures (`MultiViewBmodeVideo`, etc.) in `src/UoB/data/formats.py` based on `legacy/data/bmode.py` and `legacy/data/multiview_bmode.py`.
        - [ ] Refactor MAT file reading logic from `legacy/data/mat.py`, `legacy/data/vsx_mat.py` into `src/UoB/data/readers.py`.
        - [ ] Adapt dataset loading logic for `src/UoB/data/datasets.py`.
    - [ ] **Refactor Utilities:** Migrate geometry-related functions from `legacy/data/geo.py` and `legacy/model/*` (e.g., `lie.py`, `spatial_map.py`, `apply_pose.py`) to `src/UoB/utils/`.
    - [ ] **Refactor Registration:**
        - [ ] Migrate pose estimation logic from `legacy/model/rela_pose_est.py` to `src/UoB/registration/estimators.py`.
        - [ ] Migrate fusion logic from `legacy/model/image_fusion.py` to `src/UoB/registration/fusion.py`.
    - [ ] **Refactor Visualization:** Extract and refactor reusable functions from `legacy/visualize_*.py` into `src/UoB/visualization/`.
    - [ ] **Analyze & Place Segmentation:** Review `legacy/model/tissue_structure_coseg.py` and integrate its core logic.
    - [ ] **(Deferred)** Update imports globally.
- [ ] **Refactor Imports:** Update all import statements to reflect the new structure.
- [ ] **Setup `requirements.txt`:** Consolidate all dependencies.
- [ ] **Implement Basic Configuration:** Set up TOML loading (`tomllib` or `toml`) for dataset paths and basic parameters.
- [ ] **Refactor Preprocessing:**
    - Create a `MatConverter` class in `src/UoB/preprocessing/mat_converter.py`.
    - Make it configurable (input dir, output dir, config paths) via args or a config file.
    - Create `scripts/preprocess_data.py` to run the conversion, incorporating intermediate file checking/saving.
- [ ] **Refactor Data Loading:**
    - Define `MultiViewBmodeVideo` (and related structures) formally using `dataclasses` in `src/UoB/data/formats.py`.
    - Create a generic `UltrasoundDataset` class in `src/UoB/data/datasets.py` capable of loading the processed `.pkl` (or chosen format) files based on a config.
- **[WEB APP]** [x] **Setup Basic Web App Structure:** Initialize backend (e.g., FastAPI in `webapp/backend`) and frontend (e.g., React in `webapp/frontend`) projects, setup basic API communication (CORS, placeholder endpoints), create UI shell.
  - Initialized Next.js app (`@latest`, TypeScript, Tailwind, ESLint, App Router, src dir) in `webapp/`.
  - Fixed nested directory structure issues resulting from `create-next-app`.
  - Installed necessary type definitions (`@types/node`, `@types/react`).
  - Resolved TypeScript/Linter errors related to module resolution after dependency installation and structure correction.
- **[WEB APP]** [x] **Build Dataset Explorer:** Create UI component in the frontend to list available processed recordings (`data/processed/*`). Implement a backend API endpoint to serve recording metadata (list directories/files).
  - Created API route `webapp/src/app/api/recordings/route.ts` to scan `data/processed/` and return directory names.
  - Created React component `webapp/src/components/DatasetExplorer.tsx` to fetch and display recordings from the API.
  - Created basic main page `webapp/src/app/page.tsx` to render the `DatasetExplorer`.
  - Verified successful display of recordings in the browser. 