# Interactive Correspondence Viewer

This is an interactive visualization tool for exploring correspondence maps between different views of ultrasound images. The tool allows you to hover over points in the source image and see the corresponding points in other views in real-time.

## Features

- Interactive point selection by hovering over the source image
- Real-time correspondence visualization across all views
- Toggle between overlay and base image views
- Memory-efficient loading of correspondence matrices
- Responsive grid layout for easy comparison

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your correspondence cache files are in the correct location:
   - The correspondence files should be in a directory named `correspondence_cache`
   - Files should be named `correspondence_0.npy` through `correspondence_7.npy`

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. The source image is displayed in the first position of the grid
2. Hover your mouse over any point in the source image
3. The correspondence maps will update in real-time for all other views
4. Use the "Toggle Overlay" button to switch between showing the correspondence map as an overlay or as the base image
5. Use the "Reset View" button to reset all views to their original state

## Technical Details

- The application uses memory mapping to efficiently load large correspondence matrices
- Correspondence maps are computed on-the-fly based on the hovered point
- The visualization uses Plotly.js for smooth interactive updates
- The backend is built with Flask and handles image processing and correspondence computation

## Notes

- The application requires significant memory to load the correspondence matrices
- Performance may vary depending on your system's resources
- The correspondence maps are computed using the DINO16 model from FeatUp 