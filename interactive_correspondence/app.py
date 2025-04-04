from flask import Flask, jsonify, render_template
import numpy as np
from pathlib import Path
import os
import sys
import pickle
from PIL import Image
import torch
from torchvision import transforms

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from third_party.FeatUp.featup.util import norm
import gc

app = Flask(__name__)

class CorrespondenceViewer:
    def __init__(self, cache_dir='correspondence_cache', feature_size=224):
        # Use absolute path for cache_dir
        self.cache_dir = Path(__file__).parent / cache_dir
        self.feature_size = feature_size
        self.correspondence_files = {}
        self.memory_maps = {}
        self.frame_data = None
        self.upsampler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Cache directory: {self.cache_dir}")
        
        # Initialize upsampler
        self._init_upsampler()
        
        # Load frame data
        self._load_frame_data()
        
        # Pre-open all correspondence files with memory mapping
        self._init_correspondence_files()

    def _init_upsampler(self):
        """Initialize the FeatUp upsampler."""
        print("Loading FeatUp upsampler...")
        use_norm = True
        self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=use_norm).to(self.device)
        self.upsampler.eval()

    def _load_frame_data(self):
        """Load the frame data from pickle file."""
        print("Loading dataset...")
        current_dir = Path(__file__).parent.parent
        mvbv_path = str(current_dir / 'dataset/recording_2022-08-17_trial2-arm/1_mvbv.pkl')
        
        with open(mvbv_path, 'rb') as f:
            mvbvs = pickle.load(f)
        
        # Get first frame from LFTX views
        self.frame_data = mvbvs['lftx'].view_images[0]

    def _init_correspondence_files(self):
        """Initialize memory-mapped correspondence files."""
        print("Initializing correspondence files...")
        for i in range(8):  # 8 views
            file_path = self.cache_dir / f'correspondence_{i}.npy'
            print(f"Looking for file: {file_path}")
            if not file_path.exists():
                print(f"Warning: Correspondence file {i} not found at {file_path}")
                continue
                
            try:
                # Use the file path directly for memory mapping
                self.memory_maps[i] = np.load(str(file_path), mmap_mode='r')
                print(f"Successfully loaded correspondence file {i}")
            except Exception as e:
                print(f"Error loading correspondence file {i}: {e}")
                continue

    def get_correspondence_for_point(self, view_idx, feat_h, feat_w):
        """Get correspondence map for a specific point."""
        if view_idx not in self.memory_maps:
            return None
        corr_map = self.memory_maps[view_idx][feat_h, feat_w].copy()
        return corr_map

    def get_image_data(self, view_idx):
        """Get image data for a specific view."""
        if self.frame_data is None or view_idx >= len(self.frame_data):
            return None
            
        img = self.frame_data[view_idx]
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        return img

    def transform_point_to_feature_space(self, h, w, poi_h, poi_w):
        """Transform point coordinates to feature space."""
        # Calculate resize dimensions preserving aspect ratio
        if h > w:
            resize_h = int(self.feature_size * h / w)
            resize_w = self.feature_size
        else:
            resize_h = self.feature_size
            resize_w = int(self.feature_size * w / h)
        
        # Calculate crop offsets
        crop_y = (resize_h - self.feature_size) // 2
        crop_x = (resize_w - self.feature_size) // 2
        
        # Transform point coordinates
        feat_h = int((poi_h * resize_h / h - crop_y))
        feat_w = int((poi_w * resize_w / w - crop_x))
        
        # Clamp to valid feature space coordinates
        feat_h = max(0, min(feat_h, self.feature_size - 1))
        feat_w = max(0, min(feat_w, self.feature_size - 1))
        
        return feat_h, feat_w

    def resize_correspondence_map(self, corr_map, target_h, target_w):
        """Resize correspondence map to match target dimensions."""
        # Calculate resize dimensions preserving aspect ratio
        if target_h > target_w:
            resize_h = int(self.feature_size * target_h / target_w)
            resize_w = self.feature_size
        else:
            resize_h = self.feature_size
            resize_w = int(self.feature_size * target_w / target_h)
        
        # Calculate crop offsets
        crop_y = (resize_h - self.feature_size) // 2
        crop_x = (resize_w - self.feature_size) // 2
        
        # Create padded version at resize dimensions
        padded_map = np.zeros((resize_h, resize_w))
        padded_map[crop_y:crop_y + self.feature_size, crop_x:crop_x + self.feature_size] = corr_map
        
        # Final resize to target dimensions using PIL for better interpolation
        pil_map = Image.fromarray(padded_map)
        resized_map = np.array(pil_map.resize((target_w, target_h), Image.Resampling.BILINEAR))
        
        return resized_map

    def __del__(self):
        """Clean up file handles."""
        # No need to close files as we're using direct memory mapping
        pass

# Initialize viewer
viewer = CorrespondenceViewer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_correspondence/<int:view_idx>/<int:poi_h>/<int:poi_w>')
def get_correspondence(view_idx, poi_h, poi_w):
    # Get source image dimensions
    source_img = viewer.get_image_data(0)
    if source_img is None:
        return jsonify({'error': 'Source image not found'}), 404
    
    h, w = source_img.shape
    
    # Transform point to feature space
    feat_h, feat_w = viewer.transform_point_to_feature_space(h, w, poi_h, poi_w)
    
    # Get correspondence map
    corr_map = viewer.get_correspondence_for_point(view_idx, feat_h, feat_w)
    if corr_map is None:
        return jsonify({'error': 'Correspondence map not found'}), 404
    
    # Resize correspondence map to match image dimensions
    corr_map_resized = viewer.resize_correspondence_map(corr_map, h, w)
    
    # Normalize the correspondence map to [0, 1]
    corr_map_norm = (corr_map_resized - corr_map_resized.min()) / (corr_map_resized.max() - corr_map_resized.min() + 1e-8)
    
    return jsonify({
        'correspondence': corr_map_norm.tolist(),
        'shape': {'h': h, 'w': w}
    })

@app.route('/get_image/<int:view_idx>')
def get_image(view_idx):
    img = viewer.get_image_data(view_idx)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404
    
    # Convert to PIL Image and encode as base64
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    import base64
    from io import BytesIO
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'image': img_str
    })

if __name__ == '__main__':
    app.run(debug=True) 