# Changelog

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

### Phase 2: Feature Extraction and Visualization

10. [ ] **Integrate Feature Extractors:**
    *   Wrap foundation models (DINOv2?) and FeatUp into classes in `src/UoB/features/extractors.py` and `src/UoB/features/upsamplers.py`.
    *   Make model selection configurable (e.g., specify `dino16` via config).
11. [ ] **Refactor Visualization Code:**
    *   Move PCA logic and plotting functions from `visualize_features.py` to `src/UoB/visualization/plot_features.py`.
    *   Move correspondence computation and plotting from `visualize_similarity.py` to `src/UoB/features/matching.py` and `src/UoB/visualization/plot_correspondence.py`.
    *   Make visualization functions accept data objects (e.g., `MultiViewBmodeVideo` frames) and feature tensors.
12. [ ] **Create Feature Extraction Script:** Develop `scripts/extract_features.py` that loads data, selects a feature extractor via config, computes features, and optionally saves them.
13. [ ] **Create Visualization Script:** Develop `scripts/visualize.py` with subcommands (e.g., `visualize.py features`, `visualize.py correspondence`) driven by configs.
14. [ ] **Address Memory Issues:** Systematically apply memory management techniques (CPU/GPU transfer, `del`, `gc.collect`, `torch.cuda.empty_cache`) where needed.
15. **[WEB APP]** [~] **Visualize Raw Frames:** Enhance Dataset Explorer UI to select a recording and view individual frames (LF/HF, different views) from its `MultiViewBmodeVideo` file. Requires backend API endpoint to load/serve specific frames.
    - **Setup Python Data Service:** Created a separate FastAPI service (`webapp/data_service`) to handle `.pkl` loading.
    - **Service Endpoints:** Implemented `/ping`, `/details`, and `/frames` endpoints in the Python service.
    - **Data Loading & Processing:** Added logic to load `combined_mvbv.pkl`, extract metadata, access specific frame tensors, normalize, and convert frames to PNG using Pillow.
    - **Debugging:** Resolved issues with TOML `null` parsing, `sys.path` configuration for `src.UoB` imports, log compression saturation (`max_value`), and data orientation (transpose) during preprocessing.
    - **Next.js API:** Modified details API route and created a new frame API route to proxy requests to the Python service.
    - **Frontend UI:** Added dropdowns/input for frame selection (frequency, spatial view, frame index) in `DatasetExplorer.tsx`.
    - **Frontend Display:** Implemented image display using an `<img>` tag pointing to the Next.js frame API route.
16. **[WEB APP]** [ ] **Visualize Extracted Features:** Add UI functionality to display pre-computed feature visualizations (e.g., PCA plots) associated with a frame. Requires backend API endpoint to serve feature maps or their visualizations (e.g., load `.png` or compute/cache PCA on demand). 