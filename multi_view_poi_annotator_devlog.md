# Multi-View POI Annotator Development Log

## Understanding from Existing Code Analysis

### Dataset Structure
From analyzing `manual_reg_ui.py` and `poi_annotate.py`, I understand the multiview ultrasound dataset has the following characteristics:

#### Data Organization
- **Dataset location**: `data/processed/{recording_name}/`
- **Raw multiview data**: `combined_mvbv.pkl` - contains MVBV (Multi-View Beam-formed Volume) objects
- **Two transducer modes**: `lftx` (Low Frequency TX) and `hftx` (High Frequency TX)
- **8 views per frame**: Each transducer captures 8 different angular views
- **Frame structure**: `mvbvs[tx_mode].view_images.shape = [n_frames, n_views, height, width]`
- **Processed panoramas**: `panoramas/weighted_mean_fuser/{tx_mode}/frame_XXXX_{tx_mode}.png`

#### Coordinate System & Registration
- **View registration**: Uses theta values (rotation angles) to align adjacent views
- **Rigid link model**: Views are connected in a chain with relative rotations
- **Transformation pipeline**: 
  1. Individual views → Homography matrices → Warped images → Fused panorama
  2. Uses `RigidLink` class to calculate global poses from relative theta values
  3. Dynamic canvas creation to accommodate varying panorama sizes

### Existing POI Annotation System (poi_annotate.py)
The current POI annotator works on **panorama images** (fused views) with these features:

#### Data Structure
```python
POI = {
    'frame_name': 'frame_0000_hftx.png',  # Panorama filename
    'x': int,                             # X coordinate in panorama
    'y': int,                             # Y coordinate in panorama  
    'points': 'p1',                       # Point identifier
    'frame': int,                         # Frame number
    'video_name': 'recording_name',       # Recording identifier
    'subject': int                        # Subject ID
}
```

#### GUI Features
- Interactive point annotation with left-click to add, right-click to remove
- Frame navigation with slider and interval controls
- Reference frame loading to show previous annotations
- Incremental and batch saving capabilities

### Key Insights for Multi-View POI Annotator

#### Challenge: Cross-View Correspondence
The new tool needs to annotate **corresponding points across 8 individual views** rather than on fused panoramas. This presents several challenges:

1. **Geometric Relationships**: Points in different views show the same anatomical feature from different angles
2. **Occlusion Handling**: Some points may not be visible in all 8 views
3. **Registration Dependency**: Accurate correspondence requires knowing the geometric relationship between views
4. **Data Format**: Need to track which views contain each point

#### Required Data Structure Extension
```python
MultiViewPOI = {
    'point_id': 'p1',                    # Unique point identifier
    'frame': int,                        # Frame number
    'video_name': 'recording_name',      # Recording identifier  
    'subject': int,                      # Subject ID
    'coordinates': {                     # Per-view coordinates
        'view_0': {'x': int, 'y': int, 'visible': bool},
        'view_1': {'x': int, 'y': int, 'visible': bool},
        # ... for all 8 views
    }
}
```

#### Leverageable Components

**From manual_reg_ui.py:**
- `load_and_visualize_frame()` - loads raw MVBV data
- View pair display system with red/green overlay
- Rigid link transformation calculations
- Interactive slider controls for view selection

**From poi_annotate.py:**
- Point annotation interaction patterns (click to add/remove)
- State management for annotation sessions
- CSV save/load functionality
- Reference frame systems

#### Proposed Multi-View Workflow

1. **Frame Selection**: Use existing frame selection UI from manual_reg_ui.py
2. **View Layout**: Display all 8 views simultaneously in a grid layout
3. **Point Correspondence**: 
   - Click on first view to establish a point
   - System highlights potential corresponding regions in other views
   - User clicks on corresponding points in other views
   - Handle cases where point is not visible in some views
4. **Registration Integration**: Use existing theta values or rigid link model to predict correspondence
5. **Validation**: Show projected points overlaid on views to verify accuracy

#### Technical Architecture

**Core Components Needed:**
1. **MultiViewLoader**: Extends existing data loading to provide individual view access
2. **CorrespondenceTracker**: Manages point relationships across views  
3. **GeometricPredictor**: Uses registration data to suggest corresponding points
4. **MultiViewGUI**: 8-panel layout with cross-view interaction
5. **DataExporter**: Extended CSV format for multi-view coordinates

**Integration Points:**
- Reuse `select_recording_interactively()` for dataset selection
- Extend `AnnotationState` class for multi-view tracking
- Adapt existing matplotlib interaction patterns
- Leverage `RigidLink` transformations for correspondence prediction

This analysis provides the foundation for implementing a sophisticated multi-view POI annotation tool that builds upon the existing codebase while addressing the unique challenges of cross-view correspondence annotation.
