# Changelog

## 2025-04-28

### Planned Feature: Featurizer Config Selection Menu for Visualization Page

#### Summary
- The workspace is organized with a clear separation between frontend (Next.js in `webapp/src`), backend (FastAPI in `webapp/data_service`), and configuration files (`configs/features/`).
- The feature visualization UI is implemented in `FeatureVisualizer.tsx` and the dynamic route `visualize/[recording_id]/[frame_index]/page.tsx`.
- Feature visualization API endpoints are under `webapp/src/app/api/recordings/[recording_id]/visualize_features/[frame_index]/route.ts` (proxy) and backend logic in `webapp/data_service/routers/recordings.py`.
- Featurizer configs are TOML files in `configs/features/` (e.g., `jbu_dinov2.toml`, `jbu_dino16.toml`).
- No existing endpoint for listing featurizer configs; no config menu in the UI yet.

#### Implementation Plan

- [x] **Frontend (Next.js):**
  - [x] Add an API route to list available featurizer configs (names from `configs/features/`).
  - [x] Add a dropdown/menu to the feature visualization page (`FeatureVisualizer.tsx`) for selecting the featurizer config.
  - [x] Store the selected featurizer in React state (and optionally in the URL or local storage).
  - [x] Update visualization requests to include the selected featurizer config.
  - [x] Show a loading indicator while updating the visualization.
  - [x] Ensure the menu is accessible and does not disrupt existing UI/UX.

- [ ] **Backend (FastAPI):**
  - [ ] Add an endpoint to list available featurizer configs by reading `configs/features/`.
  - [x] Update the feature visualization endpoint to accept a featurizer config parameter and use the specified config for feature extraction/visualization.
  - [x] Validate the config name and handle errors gracefully.
  - [x] Update caching logic to account for the featurizer config parameter.

- [ ] **Integration & Testing:**
  - [ ] Test the end-to-end flow: selecting a featurizer updates the visualization as expected.
  - [ ] Test error handling for missing/invalid configs.
  - [ ] Polish the config menu UI/UX and add tooltips or descriptions if needed.

#### Notes
- The new features will be added in a modular way to avoid disrupting existing functionality.
- The plan leverages the current file structure and routing conventions for minimal friction.
- Backend support for featurizer config selection has been implemented, including validation and error handling.
- Mask handling has been improved to ensure proper resizing when feature shapes don't match.
- Frontend implementation has been completed: the API route now dynamically reads available featurizer configs from the configs/features directory, making all six featurizer options (jbu_clip, jbu_dino16, jbu_dinov2, jbu_maskclip, jbu_resnet50, jbu_vit) available in the UI dropdown.

## 2025-04-18


### alpha params for pck calculation to determine the matching correctness
### show area correspondence? -- show cosine correlation map?

### Feature Visualization & Correspondence UI Improvements

- [x] **Fix Correspondence Visualization Issues:**
  - [x] Fixed initial green star positioning to match the API response coordinates
  - [x] Fix remaining green star visibility issues across all query views
  - [x] Ensure match coordinates from API are correctly interpreted and positioned on images
  - [x] Add debugging to confirm all match results are being received and processed correctly
  - [x] Add coordinate clamping and normalization to keep green stars within visible image area
  - [x] Improve coordinate processing with better feature map coordinate detection and conversion
  - [x] Add detailed coordinate debugging to help diagnose positioning issues
  - [x] Add support for mixed coordinate systems (feature map, pixel, and percentage-based)
  - [x] Implement match confidence visualization with different star sizes and opacity levels
  - [x] Add self-match validation to verify coordinate processing accuracy
  - [x] Implement error offset correction using self-match validation results
  - [x] Add source view to match results for complete debug output
  - [x] Refine coordinate clamping to use 1% minimum instead of 10% for better accuracy
  - [x] **ABANDONED:** Client-side coordinate processing approach due to persistent issues with coordinate systems
- [x] **NEW: Phased Backend Matplotlib Correspondence Visualization:**
  - [x] **Phase 1: Research & Proof of Concept**
    - [x] Review existing PCA visualization code in `src/UoB/visualization/plot_features.py`
    - [x] Review relevant matplotlib coordinate handling and grid plotting
    - [x] Create a standalone test script that generates a simple correspondence visualization
    - [x] Test with various coordinate inputs to validate robustness
  - [x] **Phase 2: Basic Backend Implementation**
    - [x] Add a minimal endpoint in `webapp/data_service/routers/recordings.py`
    - [x] Implement basic matplotlib visualization with source view and single query view
    - [x] Test with controlled inputs before expanding to full grid
    - [x] Add comprehensive error handling and validation
  - [x] **Phase 3: Complete Backend Implementation**
    - [x] Expand to full 16-view grid visualization (8 LF + 8 HF)
    - [x] Add proper markers for source POI and matches
    - [x] Add visual indicators for match confidence
    - [x] Optimize rendering performance
    - [x] Return as PNG bytes with appropriate caching headers
  - [x] **Phase 4: Frontend Integration**
    - [x] Create API route for the new backend endpoint
    - [x] Update frontend to maintain interactive POI selection
    - [x] Add visualization request button
    - [x] Display the resulting image with appropriate loading states
    - [x] Add fallback visualization for error cases
- [x] **Improve Image Display in Correspondence Grid:**
  - [x] Normalize HF and LF image heights while maintaining aspect ratio
  - [x] Add CSS to ensure consistent image sizing across frequency types
  - [x] Set explicit heights for HF images to improve layout consistency
  - [x] Add proper loading states for images
- [x] **Fix NextJS API Error in Frame Route:**
  - [x] Update `/api/recordings/[recording_id]/frames/[frame_index]/route.ts` to properly await params
  - [x] Add error handling and logging to help diagnose issues
  - [x] Fix the same NextJS params issue in the correspondence API route
  - [x] Fix the NextJS params issue in the visualize_features API route
  - [x] Use NextJS recommended approach for handling dynamic route parameters
- [x] **Enhance User Experience:**
  - [x] Make correspondence grid images persist when changing POI
  - [x] Fix handleSourceImageClick to maintain visibility when selecting new POI
  - [x] Add stronger visual indicators with larger stars and better shadows
  - [x] Add detailed debug counts for match visibility issues
  - [x] Add visual confidence indicators for match quality
  - [x] Add blue circle indicator for self-match validation
  - [x] Improve performance with optimized image loading and caching

## 2025-04-17

### PCA Visualization Enhancement

- [x] **Integrate Image Mask into PCA:**
    - [x] Modify PCA functions (`fit_joint_pca`, `apply_pca_to_features` in `src/UoB/visualization/plot_features.py`) to accept an optional preprocessed binary mask.
    - [x] Use the mask to select relevant feature vectors for PCA fitting and transformation.
    - [x] Modify the feature visualization endpoint in `webapp/data_service/main.py` to:
        - [x] Load the `view_masks` from `MultiViewBmodeVideo` data.
        - [x] Apply the same preprocessing (e.g., `PadToSquareAndAlign`) to the mask as used for the images.
        - [x] Threshold the processed mask to make it binary.
        - [x] Pass the binary mask to the updated PCA functions.
        - [x] Update endpoint to return the generated plot as PNG bytes.
    - [x] Adjust `plot_feature_pca_comparison` to handle/display masked regions appropriately in the output visualization.
    - [x] Added local test case in `plot_features.py` to verify mask processing and PCA application.
    - [x] Updated Next.js API proxy route (`visualize_features/.../route.ts`) to handle PNG response.
    - [x] Updated frontend component (`FeatureVisualizer.tsx`) to display the single plot image.

### Phase 2 & 3: Feature Matching & Correspondence

- [x] **Refactor Matching Logic (from Task 11):**
    - [x] Migrate correspondence computation logic from `legacy/visualize_similarity.py` to `src/UoB/features/matching.py`. Define methods for computing similarity/correlation between extracted deep features. (`compute_similarity_matrix`, `find_nearest_neighbors`, `find_mutual_nearest_neighbors`, `find_k_nearest_neighbors` added and tested).
- [x] **Refactor Correspondence Visualization (from Task 11):**
    - [x] Migrate correspondence plotting logic from `legacy/visualize_similarity.py` to `src/UoB/visualization/plot_correspondence.py`. Implemented `plot_correspondences` for sparse matches and tested with example using real data.
- [x] **Develop Correspondence Visualization Script (Task 13):**
    - [x] Created `scripts/visualize.py` with a `correspondence` subcommand using `typer`.
    - [x] Implemented logic to load config, data, models, extract features, compute similarity, find k-NN matches for random POIs, run joint PCA.
    - [x] Calls `plot_correspondences` to show results on both original and PCA images.
    - [ ] (Optional) Add support for MNN matching visualization.
    - [ ] (Optional) Add support for specifying POIs via arguments instead of random sampling.
- [x] **Plan/Start Web App Correspondence Backend & Frontend (Task 17):**
    - [x] **Backend (`webapp/data_service/main.py`):
        - [x] **Define Endpoint:** Create `POST /recordings/{recording_id}/correspondence/{frame_index}`.
        - [x] **Define Request Body:** Expect Pydantic model with `source_view_index: int`, `poi_normalized: List[float]`.
        - [x] **Implement Logic:**
            - [x] Added feature caching helper `_get_or_compute_frame_features`.
            - [x] Load data, validate frame/view.
            - [x] Extract/cache features for source and all query views.
            - [x] Convert normalized POI `[y, x]` to feature map pixel coords `[r, c]`.
            - [x] Convert POI coords to flat index, isolate POI feature vector.
            - [x] Loop through query views: Compute cosine similarity between POI vector and all query vectors, find 1-NN flat index.
            - [x] Convert match flat index back to 2D coords `[match_r, match_c]`. 
            - [x] Fixed shape mismatches during feature transform application.
        - [x] **Define Response:** Return JSON `Dict[query_view_index: int, match_coords: List[int]]` (feature map coords).
    - [x] **Modularization:** Refactored `webapp/data_service/main.py` into modules (`config.py`, `cache.py`, `utils.py`, `features.py`) and routers (`routers/recordings.py`, `routers/cache.py`, `routers/misc.py`).
    - [x] **API Proxy (`webapp/src/app/api/...`):
        - [x] **Create Route:** Add `.../correspondence/[frame_index]/route.ts`.
        - [x] **Implement `POST` Handler:** Parse params/body, `fetch` backend POST endpoint, forward JSON response/errors.
    - [x] **Frontend (`webapp/src/components/...`):
        - [x] **State:** Add state for `sourceViewIndex`, `poiCoords` (normalized), `matchResults`, `isComputing`.
        - [x] **UI (Grid):** Add hover button ("Select Source") to set `sourceViewIndex`. Overlay markers based on `matchResults`.
        - [x] **UI (Interactive View):** Display selected source view, handle clicks to set `poiCoords`, display POI marker.
        - [x] **UI (Button):** Show "Compute" button when `poiCoords` is set.
        - [x] **API Call:** On button click, call proxy API, update `matchResults` state.

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
    *   [x] **Implement JBU Loading:** Implement `JointBilateralUpsampler._load_model` using `torch.hub.load("mhamilton723/FeatUp", backbone_hub_id, ...)`

## 2025-04-15

### Web App Enhancements & Optimization

- [x] **Plan/Start Web App Correspondence Backend & Frontend (Task 17):**
    - [x] **Backend (`webapp/data_service/main.py`):
        - [x] **Define Endpoint:** Create `POST /recordings/{recording_id}/correspondence/{frame_index}`.
        - [x] **Define Request Body:** Expect Pydantic model with `source_view_index: int`, `poi_normalized: List[float]`. Optional `k: int`.
        - [x] **Implement Logic:**
            - [x] Load data, validate frame/view.
            - [x] Extract features for source and all query views.
            - [x] Convert normalized POI `[y, x]` to feature map pixel coords `[r, c]`.
            - [x] Convert POI coords to flat index, isolate POI feature vector.
            - [x] Loop through query views: Compute similarity between POI vector and all query vectors, find 1-NN flat index.
            - [x] Convert match flat index back to 2D coords `[match_r, match_c]`. 
        - [x] **Define Response:** Return JSON `Dict[query_view_index: int, match_coords: List[int]]`.
    - [x] **API Proxy (`webapp/src/app/api/...`):
        - [x] **Create Route:** Add `.../correspondence/[frame_index]/route.ts`.
        - [x] **Implement `POST` Handler:** Parse params/body, `fetch` backend POST endpoint, forward JSON response/errors.
    - [x] **Frontend (`webapp/src/components/...`):
        - [x] **State:** Add state for `sourceViewIndex`, `poiCoords` (normalized), `matchResults`, `isComputing`.
        - [x] **UI (Grid):** Add hover button ("Select Source") to set `sourceViewIndex`. Overlay markers based on `matchResults`.
          - *Note:* The backend returns a single composite image (8x4 grid of plots). Plan: Use an absolute 8x4 CSS grid overlay on top of the image. Attach hover/click handlers to the overlay cells corresponding to input views (rows 0 and 2). Map cell index to logical `sourceViewIndex`. 
          - *Alternative:* If the overlay approach proves too complex/fragile, revert to using a dropdown menu or checkboxes list outside the image to select the source view.
        - [x] **UI (Interactive View):** Display selected source view, handle clicks to set `poiCoords`, display POI marker.
        - [x] **UI (Button):** Show "Compute" button when `poiCoords` is set.
        - [x] **API Call:** On button click, call proxy API, update `matchResults` state.