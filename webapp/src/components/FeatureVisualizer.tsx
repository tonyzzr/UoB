'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link'; // For back button
import { FaLocationArrow, FaStar, FaSpinner, FaExclamationTriangle, FaCheck } from 'react-icons/fa';

// Define the structure of the params prop expected from the page
interface PageParams {
  recording_id: string;
  frame_index: string;
}

interface FeatureVisualizerProps {
  params: PageParams; // Accept the params object
}

const FeatureVisualizer: React.FC<FeatureVisualizerProps> = ({ params }) => { // params might be ReactPromise, handle dynamically
  
  // State to hold the resolved params values
  const [resolvedParams, setResolvedParams] = useState<PageParams | null>(null);
  
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  
  // --- State for Interactive Correspondence ---
  const [sourceViewIndex, setSourceViewIndex] = useState<number | null>(null);
  const [poiCoords, setPoiCoords] = useState<{ x: number; y: number } | null>(null); // Storing normalized coords
  const [matchResults, setMatchResults] = useState<{ [queryViewIndex: number]: number[] } | null>(null);
  const [isComputing, setIsComputing] = useState<boolean>(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const [sourceImageDimensions, setSourceImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const interactiveImageRef = useRef<HTMLImageElement>(null); // Ref for the interactive image
  
  // Add state for debug self-match
  const [debugSelfMatch, setDebugSelfMatch] = useState<number[] | null>(null);
  
  // --- State for matplotlib-based Correspondence ---
  const [correspondenceImageUrl, setCorrespondenceImageUrl] = useState<string | null>(null);
  const [isLoadingVisImage, setIsLoadingVisImage] = useState<boolean>(false);
  const [visualizationError, setVisualizationError] = useState<string | null>(null);
  
  // --- Constants (assuming 16 views: 8 LF, 8 HF) ---
  const NUM_LF_VIEWS = 8;
  const NUM_HF_VIEWS = 8;
  const TOTAL_VIEWS = NUM_LF_VIEWS + NUM_HF_VIEWS;
  // Feature map dimensions (Hardcoded for now - adjust if needed!)
  const FEATURE_MAP_WIDTH = 16; // Example based on ViT-S/14 or /16
  const FEATURE_MAP_HEIGHT = 16;
  // Dimensions of the plots within the main visualization image
  // These might need adjustment based on the actual plot output size/layout
  const PLOT_COLS = 8;
  const PLOT_ROWS = 4; // 2 rows of images (Input + PCA) per frequency

  useEffect(() => {
    console.log("FeatureVisualizer useEffect running. Received params prop:", params);
    
    let extractedRecordingId: string | null = null;
    let extractedFrameIndex: string | null = null;
    
    // Dynamically check the structure of params
    if (params && typeof (params as any).value === 'string') { // Use type assertion for check
        try {
            const parsedValue = JSON.parse((params as any).value);
            extractedRecordingId = parsedValue.recording_id;
            extractedFrameIndex = parsedValue.frame_index;
            console.log("Parsed params.value:", { extractedRecordingId, extractedFrameIndex });
        } catch (e) {
            console.error("Error parsing params.value JSON:", e);
            setError("Failed to parse routing parameters.");
            setIsLoading(false);
            return;
        }
    } else if (params && params.recording_id && params.frame_index) {
        // Handle case where params is already the plain object
        extractedRecordingId = params.recording_id;
        extractedFrameIndex = params.frame_index;
        console.log("Using params directly:", { extractedRecordingId, extractedFrameIndex });
    }

    // Set resolved state if we extracted valid values and they differ from current state
    if (extractedRecordingId && extractedFrameIndex && 
        (!resolvedParams || 
         resolvedParams.recording_id !== extractedRecordingId || 
         resolvedParams.frame_index !== extractedFrameIndex)) {
         setResolvedParams({ recording_id: extractedRecordingId, frame_index: extractedFrameIndex });
    }

    // Proceed only if params have been successfully resolved into state
    if (!resolvedParams) {
        console.log("FeatureVisualizer useEffect: Waiting for resolved recordingId/frameIndex...");
        setIsLoading(true); // Ensure loading is true while waiting
        return; 
    }
    
    setIsLoading(true);
    setError(null);
    setImageUrl(null); // Reset image URL when props change
    
    try {
      const apiUrl = `/api/recordings/${resolvedParams.recording_id}/visualize_features/${resolvedParams.frame_index}`;
      console.log(`Setting image URL to: ${apiUrl}`);
      setImageUrl(apiUrl); // Set the URL, loading/error handled by img tag
      
      // No need to throw error here anymore, handled by the guard clause above
      
      // Set loading to false *after* successfully setting the URL
      setIsLoading(false);
      
    } catch (err: any) {
      console.error("Error setting up image URL:", err);
      setError(err.message || "An unknown error occurred.");
      setImageUrl(null);
      setIsLoading(false); // Set loading false on setup error
    }
    // Note: setIsLoading(false) will now primarily be handled by the image's onLoad/onError
    // or if the initial setup fails.
  }, [params, resolvedParams]); // Depend on params prop and resolved state

  // --- Event Handlers ---
  const handleSourceImageLoad = (event: React.SyntheticEvent<HTMLImageElement, Event>) => {
    // Get intrinsic dimensions of the loaded interactive source image
    const { naturalWidth, naturalHeight } = event.currentTarget;
    setSourceImageDimensions({ width: naturalWidth, height: naturalHeight });
    console.log(`Interactive source image loaded with dimensions: ${naturalWidth}x${naturalHeight}`);
  };
  
  const handleSourceImageClick = (event: React.MouseEvent<HTMLImageElement>) => {
    if (!sourceImageDimensions) {
        console.warn("Cannot set POI: Source image dimensions not yet available.");
        return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    // Calculate normalized coordinates
    const normX = clickX / sourceImageDimensions.width;
    const normY = clickY / sourceImageDimensions.height;
    
    // Clamp coordinates to [0, 1]
    const clampedNormX = Math.max(0, Math.min(1, normX));
    const clampedNormY = Math.max(0, Math.min(1, normY));
    
    console.log(`Image click at (${clickX.toFixed(1)}, ${clickY.toFixed(1)}), Normalized: (${clampedNormY.toFixed(3)}, ${clampedNormX.toFixed(3)})`);
    
    setPoiCoords({ x: clampedNormX, y: clampedNormY });
    // Don't clear match results to maintain grid visibility between POI changes
    setComputeError(null); // Clear previous error
  };
  
  // Function to help debug coordinate values in the UI
  const debugCoordinateToString = (coord: number[]): string => {
    if (!coord || !Array.isArray(coord) || coord.length !== 2) return "Invalid";
    return `[${coord[0].toFixed(1)}, ${coord[1].toFixed(1)}]`;
  };

  // Function to process and normalize match coordinates, optionally applying error correction
  const processMatchCoordinates = (coords: number[], applyErrorCorrection: boolean = false): number[] => {
    if (!coords || !Array.isArray(coords) || coords.length !== 2) {
      return [50, 50]; // Default to center if invalid
    }
    
    let [y, x] = coords;
    
    // IMPORTANT: The API returns coordinates in different formats for different views:
    // 1. Feature map coordinates: Small integers 0-16 (e.g., [0, 5])
    // 2. Pixel coordinates: Larger values like [0, 213]
    
    // For all coordinates, we want to ensure they are valid percentages (0-100)
    
    // Case 1: Small feature map coordinates (typically 0-16)
    if (Number.isInteger(x) && Number.isInteger(y) && 
        x >= 0 && x <= 16 && y >= 0 && y <= 16) {
      console.log(`[Frontend] Converting feature map coordinates: [${y}, ${x}] to percentages`);
      // Convert from feature map coords to percentages (0-100%)
      x = (x / FEATURE_MAP_WIDTH) * 100;
      y = (y / FEATURE_MAP_HEIGHT) * 100;
    } 
    // Case 2: Large pixel coordinates
    else if (x > 100 || y > 100) {
      console.log(`[Frontend] Converting large pixel coordinates: [${y}, ${x}] to percentages`);
      // If these are actual pixel values, we need to convert them to percentages
      // Assuming source images are approximately 300px wide at most, and stars should be in 0-100%
      x = Math.min(99, (x / 300) * 100);
      y = Math.min(99, (y / 300) * 100);
    }
    
    // Special case handling for exact 0,0 values (likely failed matches)
    if (x === 0 && y === 0) {
      console.log(`[Frontend] Adjusting exact 0,0 coordinate to be just visible at 1%`);
      // Make exact 0,0 coordinates just barely visible
      x = 1;
      y = 1;
    }
    
    // Ensure coordinates are in 0-100% range for safety
    x = Math.max(0, Math.min(100, x));
    y = Math.max(0, Math.min(100, y));
    
    // Use a very small minimum (1%) for visibility without distorting positions
    // This allows points to be very close to the edge while still being visible
    if (x < 1) x = 1;
    if (y < 1) y = 1;
    if (x > 99) x = 99;
    if (y > 99) y = 99;
    
    return [y, x];
  };

  // Update handleComputeCorrespondence to include visualization request
  const handleComputeCorrespondence = async () => {
    if (!poiCoords || sourceViewIndex === null) return;
    
    setIsComputing(true);
    setComputeError(null);
    setVisualizationError(null);
    
    // Use resolvedParams for the API call
    if (!resolvedParams) {
        console.error("Cannot compute correspondence: resolvedParams not available.");
        setComputeError("Cannot compute correspondence: parameters not resolved.");
        setIsComputing(false);
        return;
    }

    const requestBody = {
      source_view_index: sourceViewIndex,
      poi_normalized: [poiCoords.y, poiCoords.x] // Send as [y, x]
    };
    
    // Step 1: Call the regular correspondence API for the JSON data (for debugging)
    console.log(`[Frontend] Sending correspondence request for match data...`);
    
    try {
      const apiUrl = `/api/recordings/${resolvedParams.recording_id}/correspondence/${resolvedParams.frame_index}`;
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      const data = await response.json();

      if (!response.ok) {
        console.error("[Frontend] Correspondence API Error:", data);
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }
      
      console.log("[Frontend] Received match results (raw):", data);
      
      // Process the data to ensure it's in the right format
      if (data && typeof data === 'object') {
        const formattedResults: {[key: number]: number[]} = {};
        
        // Calculate self-match for debug purposes
        // Create a synthetic coordinate in the same format as API responses
        const originalPoiPercentages = [poiCoords.y * 100, poiCoords.x * 100];
        // Process it through the same coordinate processing pipeline as other matches
        const processedSelfMatch = processMatchCoordinates(originalPoiPercentages);
        
        // Calculate error offset between original POI and processed coordinates
        const errorOffsetY = processedSelfMatch[0] - originalPoiPercentages[0];
        const errorOffsetX = processedSelfMatch[1] - originalPoiPercentages[1];
        
        // Add the self-match to the results so it appears in debug output
        formattedResults[sourceViewIndex] = originalPoiPercentages;
        
        setDebugSelfMatch(processedSelfMatch);
        console.log(`[Frontend] Debug self-match: Original POI %: [${originalPoiPercentages[0].toFixed(2)}, ${originalPoiPercentages[1].toFixed(2)}]`);
        console.log(`[Frontend] Debug self-match: Processed: [${processedSelfMatch[0].toFixed(2)}, ${processedSelfMatch[1].toFixed(2)}]`);
        console.log(`[Frontend] Error offset: [${errorOffsetY.toFixed(2)}, ${errorOffsetX.toFixed(2)}]`);
        
        // Process each key-value pair from the API response
        Object.entries(data).forEach(([viewIndex, coords]) => {
          const viewIdx = parseInt(viewIndex, 10);
          
          // Check if coords is valid (should be a 2-element array [y, x])
          if (Array.isArray(coords) && coords.length === 2) {
            // Process match coordinates
            const normalized = processMatchCoordinates(coords as number[]);
            
            // Apply error offset correction (subtract the error)
            const corrected = [
              Math.max(10, Math.min(90, normalized[0] - errorOffsetY)),
              Math.max(10, Math.min(90, normalized[1] - errorOffsetX))
            ];
            
            formattedResults[viewIdx] = corrected;
            console.log(`[Frontend] Processed match for view ${viewIdx}: Raw: [${(coords as number[])[0]}, ${(coords as number[])[1]}], ` +
                       `Normalized: [${normalized[0]}, ${normalized[1]}], ` +
                       `Corrected: [${corrected[0]}, ${corrected[1]}]`);
          } else {
            console.warn(`[Frontend] Invalid match format for view ${viewIdx}:`, coords);
          }
        });
        
        setMatchResults(formattedResults);
        console.log("[Frontend] Processed match results:", formattedResults);
        
        // Step 2: Get the visualization image
        await fetchCorrespondenceVisualization(requestBody);
      } else {
        console.error("[Frontend] Invalid data format from API:", data);
        throw new Error("Invalid data format received from API");
      }
      
    } catch (err: any) {
      console.error("[Frontend] Error computing correspondence:", err);
      setComputeError(err.message || "An unknown error occurred while computing matches.");
    } finally {
      setIsComputing(false);
    }
  };
  
  // New function to fetch the correspondence visualization
  const fetchCorrespondenceVisualization = async (requestBody: any) => {
    setIsLoadingVisImage(true);
    setVisualizationError(null);
    
    try {
      const visApiUrl = `/api/recordings/${resolvedParams!.recording_id}/visualize_correspondence/${resolvedParams!.frame_index}`;
      console.log(`[Frontend] Sending request for visualization to ${visApiUrl}`, requestBody);
      
      // Create a unique cache-busting URL parameter
      const timestamp = new Date().getTime();
      
      // Make a POST request but handle the response as a blob
      const response = await fetch(visApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        // Try to parse error as JSON
        try {
          const errorData = await response.json();
          throw new Error(errorData.error || `Visualization request failed with status ${response.status}`);
        } catch (e) {
          // If not JSON, use text or status
          const errorText = await response.text();
          throw new Error(errorText || `Visualization request failed with status ${response.status}`);
        }
      }
      
      // Get the response as a blob
      const blob = await response.blob();
      
      // Create an object URL from the blob
      const imageUrl = URL.createObjectURL(blob);
      setCorrespondenceImageUrl(imageUrl);
      console.log("[Frontend] Received visualization image");
      
    } catch (err: any) {
      console.error("[Frontend] Error fetching correspondence visualization:", err);
      setVisualizationError(err.message || "An unknown error occurred while fetching visualization.");
    } finally {
      setIsLoadingVisImage(false);
    }
  };

  // --- Rendering Logic ---
  const renderInteractiveSourceView = () => {
    if (sourceViewIndex === null) return null;
    
    // Determine freq and view index within freq
    const isHf = sourceViewIndex >= NUM_LF_VIEWS;
    const freq = isHf ? 'hftx' : 'lftx';
    const viewInFreq = isHf ? sourceViewIndex - NUM_LF_VIEWS : sourceViewIndex;
    
    // Use resolvedParams for generating image URL
    if (!resolvedParams) return <p>Loading parameters...</p>; // Should not happen if sourceViewIndex is set
    const sourceImageUrl = `/api/recordings/${resolvedParams.recording_id}/frames/${resolvedParams.frame_index}?freq=${freq}&view=${viewInFreq}`;
    
    return (
      <div className="mb-8 p-6 border border-blue-300 bg-blue-50 rounded-lg shadow-md">
        <h3 className="text-xl font-medium text-gray-800 mb-4">Interactive Source View ({freq.toUpperCase()} View {viewInFreq})</h3>
        <div className="flex flex-col md:flex-row items-start gap-6">
          <div className="relative inline-block border border-gray-300 p-2 bg-white" style={{ maxWidth: '400px' }}>
            <img 
              ref={interactiveImageRef}
              src={sourceImageUrl} 
              alt={`Source View ${freq.toUpperCase()} ${viewInFreq}`} 
              className="block max-w-full h-auto cursor-crosshair" 
              onClick={handleSourceImageClick}
              onLoad={handleSourceImageLoad} // Get dimensions on load
              onError={() => setComputeError("Failed to load interactive source image.")}
            />
            {/* Display POI Marker */}  
            {poiCoords && sourceImageDimensions && (
              <FaStar 
                className="absolute text-red-600 fill-red-600 text-2xl pointer-events-none" 
                style={{
                  left: `${poiCoords.x * 100}%`, 
                  top: `${poiCoords.y * 100}%`, 
                  transform: 'translate(-50%, -50%)', // Center the star
                  color: '#dc2626', // Explicit hex color for red-600
                  fill: '#dc2626', // Explicit hex color for red-600
                }}
              />
            )}
          </div>
          <div className="flex-1 flex flex-col justify-between">
            <div>
                <p className="text-sm text-gray-600 mb-3">Click on the image above to select a Point of Interest (POI).</p>
                {poiCoords && (
                    <p className="text-sm text-green-700 mb-3"><FaCheck className="inline mr-1"/> POI Selected at ({poiCoords.y.toFixed(3)}, {poiCoords.x.toFixed(3)})</p>
                )}
            </div>
            <button 
              onClick={handleComputeCorrespondence}
              disabled={!poiCoords || isComputing}
              className={`mt-4 w-full md:w-auto px-6 py-2 rounded-md text-white font-medium transition-colors duration-150 
                          ${!poiCoords ? 'bg-gray-400 cursor-not-allowed' : 
                           isComputing ? 'bg-yellow-500 cursor-wait' : 
                           'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'}`}
            >
              {isComputing ? (
                <><FaSpinner className="animate-spin inline mr-2" /> Computing...</>
              ) : (
                <><FaLocationArrow className="inline mr-2" /> Compute Correspondences</>
              )}
            </button>
            {computeError && (
                <p className="text-sm text-red-600 mt-2 flex items-center"><FaExclamationTriangle className="mr-1"/> Error: {computeError}</p>
            )}
          </div>
        </div>
      </div>
    );
  }
  
  const renderFeatureGrid = () => {
    // Render the main visualization image and overlay interactive elements
    if (!imageUrl) return null;

    return (
        <div className="border border-gray-300 shadow-md rounded overflow-hidden">
            {/* The actual visualization image */} 
            <img 
                src={imageUrl} 
                alt={`Feature Visualization for ${resolvedParams?.recording_id}, frame ${resolvedParams?.frame_index}`} 
                className="block w-full h-auto"
                onLoad={() => setIsLoading(false)} 
                onError={(e) => {
                    console.error("Error loading visualization image from API route:", e);
                    setError("Failed to load the visualization image from the server.");
                    setImageUrl(null);
                    setIsLoading(false);
                }}
            />

            {/* Source View Selection Table */}
            <div className="p-4 bg-gray-50">
                <h4 className="text-lg font-medium text-gray-700 mb-2">Select a Source View:</h4>
                <p className="text-sm text-gray-600 mb-3">Click on any input view below to select it as your source view for correspondence computation.</p>
                
                {/* LF Views (Row 1) */}
                <div className="mb-1 font-medium text-gray-700">Low-Frequency Views:</div>
                <div className="grid grid-cols-8 gap-2 mb-4">
                    {Array.from({ length: NUM_LF_VIEWS }).map((_, index) => (
                        <button
                            key={`lf-${index}`}
                            className={`p-2 border rounded text-center transition-colors
                                      hover:bg-blue-100 hover:border-blue-300 focus:outline-none
                                      ${sourceViewIndex === index ? 
                                        'bg-green-100 border-green-500 font-medium' : 
                                        'bg-white border-gray-300'}`}
                            onClick={() => {
                                console.log(`Selected LF ${index}`);
                                setSourceViewIndex(index);
                                // Only reset match results if changing source view, not POI
                                // This allows changing POI while keeping the grid displayed
                                setPoiCoords(null);
                                // Keep matchResults to maintain grid visibility
                                setComputeError(null);
                                setSourceImageDimensions(null);
                            }}
                        >
                            <div className="text-sm">LF {index}</div>
                            {/* Show match indicator if match exists */}
                            {matchResults && matchResults[index] && 
                             Array.isArray(matchResults[index]) && matchResults[index].length === 2 && (
                                <div className="mt-1">
                                    <FaStar className="inline text-blue-500 mx-auto" />
                                </div>
                            )}
                        </button>
                    ))}
                </div>
                
                {/* HF Views (Row 2) */}
                <div className="mb-1 font-medium text-gray-700">High-Frequency Views:</div>
                <div className="grid grid-cols-8 gap-2">
                    {Array.from({ length: NUM_HF_VIEWS }).map((_, index) => {
                        const viewIndex = index + NUM_LF_VIEWS;
                        return (
                            <button
                                key={`hf-${index}`}
                                className={`p-2 border rounded text-center transition-colors
                                          hover:bg-blue-100 hover:border-blue-300 focus:outline-none
                                          ${sourceViewIndex === viewIndex ? 
                                            'bg-green-100 border-green-500 font-medium' : 
                                            'bg-white border-gray-300'}`}
                                onClick={() => {
                                    console.log(`Selected HF ${index}`);
                                    setSourceViewIndex(viewIndex);
                                    // Only reset match results if changing source view, not POI
                                    // This allows changing POI while keeping the grid displayed
                                    setPoiCoords(null);
                                    // Keep matchResults to maintain grid visibility
                                    setComputeError(null);
                                    setSourceImageDimensions(null);
                                }}
                            >
                                <div className="text-sm">HF {index}</div>
                                {/* Show match indicator if match exists */}
                                {matchResults && matchResults[viewIndex] && 
                                 Array.isArray(matchResults[viewIndex]) && matchResults[viewIndex].length === 2 && (
                                    <div className="mt-1">
                                        <FaStar className="inline text-blue-500 mx-auto" />
                                    </div>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
  }

  // Replace the complex client-side rendering with simpler image display
  const renderCorrespondenceGrid = () => {
    if (!matchResults && !correspondenceImageUrl) return null;
    
    return (
      <div className="mt-8 border border-gray-300 shadow-md rounded overflow-hidden bg-white">
        <div className="bg-gray-50 py-3 px-4 border-b border-gray-300">
          <h3 className="text-lg font-medium text-gray-800">Computed Correspondences</h3>
          <p className="text-sm text-gray-600">
            Green stars show the matching points across all 16 views. 
            The source view (LF {sourceViewIndex! < NUM_LF_VIEWS ? sourceViewIndex : ""}
            {sourceViewIndex! >= NUM_LF_VIEWS ? "HF " + (sourceViewIndex! - NUM_LF_VIEWS) : ""}) 
            is highlighted in green.
          </p>
        </div>

        <div className="p-4">
          {isLoadingVisImage ? (
            <div className="flex items-center justify-center py-8">
              <FaSpinner className="animate-spin text-blue-500 text-2xl mr-3" />
              <p className="text-gray-600">Loading visualization...</p>
            </div>
          ) : visualizationError ? (
            <div className="bg-red-50 border border-red-200 rounded p-4 text-red-700">
              <p className="font-medium">Error loading visualization</p>
              <p className="text-sm">{visualizationError}</p>
            </div>
          ) : correspondenceImageUrl ? (
            <div className="flex justify-center">
              <img 
                src={correspondenceImageUrl} 
                alt="Correspondence Visualization" 
                className="max-w-full h-auto border rounded shadow-sm"
                onError={() => setVisualizationError("Failed to load visualization image.")}
              />
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No visualization available.
            </div>
          )}
          
          {/* Small debugging section */}
          <div className="mt-4 text-xs text-gray-500">
            <p>Processed {matchResults ? Object.keys(matchResults).length : 0} matches. 
               {matchResults && Object.values(matchResults).filter(coords => coords[0] > 0 && coords[1] > 0).length} valid matches.</p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="text-xl font-semibold text-gray-900 hover:text-blue-600">
                &larr; Back to Explorer
              </Link>
            </div>
            <div className="flex items-center">
              <span className="text-lg font-medium text-gray-700">Feature Visualization</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="bg-white mx-auto max-w-6xl shadow-sm rounded-lg p-8">
          <h2 className="text-2xl font-medium text-center text-gray-900 mb-4">
            Visualizing Features for:
          </h2>
          <div className="text-center mb-8">
            <p>Recording: <span className="font-semibold">{resolvedParams?.recording_id || '...'}</span></p>
            <p>Frame: <span className="font-semibold">{resolvedParams?.frame_index || '...'}</span></p>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center space-x-2 text-gray-600 py-20">
              <svg className="animate-spin h-8 w-8" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-lg">Loading visualization...</p>
            </div>
          )}

          {error && (
            <div className="border-2 border-dashed border-red-300 bg-red-50 rounded-lg p-12 text-center">
              <p className="text-red-600 font-semibold text-lg">Error Loading Features:</p>
              <p className="text-red-500 mt-2">{error}</p>
            </div>
          )}

          {!isLoading && !error && imageUrl && (
            <div className="">
                <h3 className="text-xl font-medium text-gray-800 mb-4 text-center">Combined Feature Visualization (16 Views)</h3>
                <p className="text-sm text-gray-600 text-center mb-2">Click on a view below to select it as your source view for computing correspondences.</p>
                
                {/* Step 1: Main feature visualization grid */}
                {renderFeatureGrid()}
                
                {/* Step 2: POI selection section - now AFTER the feature grid */}
                {renderInteractiveSourceView()}
                
                {/* Step 3: Show correspondence results across all 16 views */}
                {renderCorrespondenceGrid()}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default FeatureVisualizer; 