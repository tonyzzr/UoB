'use client'; // Required for components using hooks like useState, useEffect

import React, { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation'; // Import the router hook

interface RecordingsResponse {
  recordings?: string[];
  error?: string;
}

// Define interface for the details API response
interface RecordingDetailsResponse {
  id?: string;
  // message?: string; // No longer returned by Python service
  error?: string;
  pkl_path?: string; // Field returned by Python service
  frame_count?: number | string; // Field returned by Python service (might be 'N/A')
  // available_views?: string[] | string; // Old field name
  num_spatial_views?: number | string; // New field from Python service
  available_freq_views?: string[]; // New field from Python service
}

// --- TEMPORARY HELPER to hide date --- 
const formatRecordingName = (fullName: string): string => {
    // Simple regex to remove the YYYY-MM-DD part
    // To revert, simply remove the usage of this function
    return fullName.replace(/_\d{4}-\d{2}-\d{2}/, '');
};

const DatasetExplorer: React.FC = () => {
  const router = useRouter(); // Initialize the router
  const [recordings, setRecordings] = useState<string[]>([]);
  const [selectedRecording, setSelectedRecording] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // State for recording details
  const [recordingDetails, setRecordingDetails] = useState<RecordingDetailsResponse | null>(null);
  const [isDetailsLoading, setIsDetailsLoading] = useState<boolean>(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);

  // State for frame selection - only need frame index now
  const [selectedFrame, setSelectedFrame] = useState<number>(0);
  // const [selectedFreq, setSelectedFreq] = useState<string | null>(null); // Removed
  // const [selectedSpatialView, setSelectedSpatialView] = useState<number>(0); // Removed

  // Effect to fetch the list of recordings (runs once)
  useEffect(() => {
    const fetchRecordings = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/recordings');
        const data: RecordingsResponse = await response.json();

        if (!response.ok) {
          throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        if (data.recordings) {
          setRecordings(data.recordings);
        } else {
            setRecordings([]); // Handle case where recordings array might be missing
        }
      } catch (err: any) {
        console.error("Failed to fetch recordings:", err);
        setError(err.message || "An unknown error occurred");
        setRecordings([]); // Clear recordings on error
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecordings();
  }, []); // Empty dependency array ensures this runs only once on mount

  // Effect to fetch details and reset frame selection when recording changes
  useEffect(() => {
    if (!selectedRecording) {
      setRecordingDetails(null);
      setSelectedFrame(0); // Only reset frame
      // setSelectedFreq(null); // Removed
      // setSelectedSpatialView(0); // Removed
      return;
    }

    const fetchRecordingDetails = async () => {
      setIsDetailsLoading(true);
      setDetailsError(null);
      setRecordingDetails(null);
      try {
        const response = await fetch(`/api/recordings/${selectedRecording}`);
        const data: RecordingDetailsResponse = await response.json();

        if (!response.ok) {
          throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        setRecordingDetails(data);
        // Reset frame selections on new details load
        setSelectedFrame(0); // Reset frame to 0
      } catch (err: any) {
        console.error(`Failed to fetch details for ${selectedRecording}:`, err);
        setDetailsError(err.message || "An unknown error occurred fetching details");
        setRecordingDetails(null); // Clear details on error
      } finally {
        setIsDetailsLoading(false);
      }
    };

    fetchRecordingDetails();
  }, [selectedRecording]); // Dependency array: runs when selectedRecording changes

  // --- Memoized values for UI controls ---
  const frameCount = useMemo(() => typeof recordingDetails?.frame_count === 'number' ? recordingDetails.frame_count : 0, [recordingDetails]);
  const spatialViewCount = useMemo(() => typeof recordingDetails?.num_spatial_views === 'number' ? recordingDetails.num_spatial_views : 0, [recordingDetails]);
  const freqViews = useMemo(() => recordingDetails?.available_freq_views || [], [recordingDetails]); // Should be ['lftx', 'hftx'] or similar

  // --- Generate Image URLs for the Grid ---
  const imageGridUrls = useMemo(() => {
    if (!selectedRecording || !recordingDetails || freqViews.length === 0 || spatialViewCount === 0 || frameCount === 0) {
      return [];
    }

    const urls: { freq: string; view: number; url: string }[] = [];
    freqViews.forEach(freq => {
        for (let view = 0; view < spatialViewCount; view++) {
            urls.push({
                freq,
                view,
                url: `/api/recordings/${selectedRecording}/frames/${selectedFrame}?freq=${freq}&view=${view}`
            });
        }
    });
    // Sort to ensure LF is always before HF if present
    urls.sort((a, b) => {
        if (a.freq.toLowerCase().includes('lf') && !b.freq.toLowerCase().includes('lf')) return -1;
        if (!a.freq.toLowerCase().includes('lf') && b.freq.toLowerCase().includes('lf')) return 1;
        return a.view - b.view; // Then sort by view number
    });
    return urls;
  }, [selectedRecording, selectedFrame, recordingDetails, freqViews, spatialViewCount, frameCount]);

  // --- Grid Component ---
  const ImageGrid = () => {
    if (imageGridUrls.length === 0) {
      return <p className="text-gray-500 text-center col-span-full">Select a recording to load images.</p>;
    }

    // Assuming 2 frequencies (LF, HF) and 8 spatial views per freq
    const lfImages = imageGridUrls.filter(img => img.freq.toLowerCase().includes('lf'));
    const hfImages = imageGridUrls.filter(img => img.freq.toLowerCase().includes('hf'));

    return (
      <div className="grid grid-cols-8 gap-2 w-full">
        {/* LF Row */}
        {lfImages.length > 0 && <div className="col-span-8 text-center font-medium text-gray-700 mb-2">{freqViews.find(f => f.toLowerCase().includes('lf')) || 'Low Freq'} Views 0-7</div>}
        {lfImages.map(({ url, freq, view }) => (
            <div key={url} className="aspect-w-1 aspect-h-4 border border-gray-200 rounded-md overflow-hidden flex items-center justify-center bg-white">
                <img
                    src={url}
                    alt={`Frame ${selectedFrame}, Freq ${freq}, View ${view}`}
                    className="max-w-full max-h-full object-contain"
                    loading="lazy"
                    onError={(e) => {
                        e.currentTarget.src = ''; // Clear src on error
                        e.currentTarget.alt = `Error loading ${freq} view ${view}`;
                    }}
                />
            </div>
        ))}
        {lfImages.length === 0 && <div className="col-span-8 h-20 bg-white rounded-md flex items-center justify-center text-gray-400">No LF images found/loaded</div>}

        {/* HF Row */}
        {hfImages.length > 0 && <div className="col-span-8 text-center font-medium text-gray-700 mt-4 mb-2">{freqViews.find(f => f.toLowerCase().includes('hf')) || 'High Freq'} Views 0-7</div>}
        {hfImages.map(({ url, freq, view }) => (
            <div key={url} className="aspect-w-1 aspect-h-4 border border-gray-200 rounded-md overflow-hidden flex items-center justify-center bg-white">
                <img
                    src={url}
                    alt={`Frame ${selectedFrame}, Freq ${freq}, View ${view}`}
                    className="max-w-full max-h-full object-contain"
                    loading="lazy"
                    onError={(e) => {
                        e.currentTarget.src = ''; // Clear src on error
                        e.currentTarget.alt = `Error loading ${freq} view ${view}`;
                    }}
                />
            </div>
        ))}
        {hfImages.length === 0 && <div className="col-span-8 h-20 bg-white rounded-md flex items-center justify-center text-gray-400">No HF images found/loaded</div>}
      </div>
    );
  };

  // --- Handle Navigation to Feature Visualization --- 
  const handleViewFeaturesClick = () => {
    if (selectedRecording && recordingDetails && typeof selectedFrame === 'number') {
      router.push(`/visualize/${selectedRecording}/${selectedFrame}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Navigation Bar - Styled to match Item Organizer */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <span className="text-xl font-semibold text-gray-900">Ultrasound Data Explorer</span>
            </div>
            <div className="flex items-center">
              <button className="ml-4 bg-[#4285f4] hover:bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium">
                Sign In
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content - Lighter background matching Item Organizer */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {/* Center Content Container with white background */}
        <div className="bg-white mx-auto max-w-5xl shadow-sm rounded-lg p-8">
          {/* Recordings Section */}
          <div className="mb-8">
            <h2 className="text-2xl font-medium text-center text-gray-900 mb-6">Processed Recordings</h2>
            {isLoading && (
              <div className="flex items-center justify-center space-x-2 text-gray-600">
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p>Loading recordings...</p>
              </div>
            )}
            {error && <p className="text-red-500 bg-red-50 p-2 rounded-md">Error: {error}</p>}
            {!isLoading && !error && (
              recordings.length > 0 ? (
                <div className="border border-gray-200 rounded-md overflow-hidden mx-auto max-w-lg">
                  <ul className="divide-y divide-gray-200 max-h-40 overflow-y-auto custom-scrollbar">
                    {recordings.map((rec) => (
                      <li
                        key={rec}
                        className={`
                          px-4 py-3
                          text-gray-700
                          cursor-pointer truncate
                          transition-colors duration-150
                          ${selectedRecording === rec
                            ? 'bg-[#f1f8ff] text-[#4285f4] font-medium'
                            : 'hover:bg-gray-50'}
                        `}
                        onClick={() => setSelectedRecording(rec)}
                        title={rec}
                      >
                        {formatRecordingName(rec)}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-gray-500 italic text-center">No processed recordings found.</p>
              )
            )}
          </div>

          {/* Details and Controls Section */}
          {selectedRecording && (
            <div className="mb-8">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-medium text-gray-900">
                  Details & Controls
                </h3>
                <span className="inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium bg-[#e8f0fe] text-[#4285f4]" title={selectedRecording}>
                  {formatRecordingName(selectedRecording)}
                </span>
              </div>
              
              {isDetailsLoading && (
                <div className="flex items-center justify-center space-x-2 text-gray-600">
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <p>Loading details...</p>
                </div>
              )}
              {detailsError && <p className="text-red-500 bg-red-50 p-2 rounded-md">Error loading details: {detailsError}</p>}
              {recordingDetails && !isDetailsLoading && !detailsError && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
                  {/* Metadata Display */}
                  <div className="md:col-span-1 space-y-3 p-4 bg-gray-50 rounded-md">
                    <h4 className="font-medium text-gray-800 mb-2">Recording Information</h4>
                    <p className="text-sm text-gray-600"><span className="font-medium text-gray-700">Path:</span> {recordingDetails.pkl_path || 'N/A'}</p>
                    <p className="text-sm text-gray-600"><span className="font-medium text-gray-700">Frames:</span> {frameCount}</p>
                    <p className="text-sm text-gray-600">
                      <span className="font-medium text-gray-700">Freq Views:</span> {freqViews.join(', ') || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-600">
                      <span className="font-medium text-gray-700">Spatial Views (per Freq):</span> {spatialViewCount}
                    </p>
                  </div>

                  {/* Frame Slider Control */}
                  <div className="md:col-span-2 p-4 bg-gray-50 rounded-md">
                    <h4 className="font-medium text-gray-800 mb-2">Frame Selection</h4>
                    <div className="flex items-center space-x-2 mb-3">
                      <label htmlFor="frameSlider" className="text-sm font-medium text-gray-700">
                        Current Frame:
                      </label>
                      <span className="font-bold text-[#4285f4] text-xl">{selectedFrame}</span>
                      <span className="text-gray-500">/ {frameCount > 0 ? frameCount - 1 : 0}</span>
                    </div>
                    
                    {/* Enhanced slider styling */}
                    <div className="relative pt-1">
                      <input
                        type="range"
                        id="frameSlider"
                        min="0"
                        max={frameCount > 0 ? frameCount - 1 : 0}
                        value={selectedFrame}
                        onChange={(e) => setSelectedFrame(Number(e.target.value))}
                        disabled={frameCount <= 1}
                        className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 accent-[#4285f4] focus:outline-none focus:ring-2 focus:ring-[#4285f4] focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                          WebkitAppearance: 'none',
                        }}
                      />
                      {/* Slider labels */}
                      <div className="w-full flex justify-between text-xs text-gray-500 px-1 mt-1">
                        <span>0</span>
                        {frameCount > 20 && <span>{Math.floor(frameCount / 4)}</span>}
                        {frameCount > 10 && <span>{Math.floor(frameCount / 2)}</span>}
                        {frameCount > 20 && <span>{Math.floor(frameCount * 3 / 4)}</span>}
                        <span>{frameCount > 0 ? frameCount - 1 : 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* --- View Features Button --- */} 
              {recordingDetails && !isDetailsLoading && frameCount > 0 && (
                  <div className="flex justify-center mt-4">
                      <button 
                          onClick={handleViewFeaturesClick}
                          disabled={!selectedRecording || isDetailsLoading} // Disable if no recording or details are loading
                          className="bg-[#4285f4] hover:bg-blue-600 text-white px-6 py-2 rounded-md text-sm font-medium shadow-sm transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                          View Deep Features for Frame {selectedFrame}
                      </button>
                  </div>
              )}
            </div>
          )}

          {/* Image Grid Section */}
          {selectedRecording && recordingDetails && (
            <div>
              <h3 className="text-xl font-medium text-gray-900 mb-6 text-center">Multi-View Frame Visualizer</h3>
              <ImageGrid />
            </div>
          )}
          
          {/* Empty State */}
          {(!selectedRecording || !recordingDetails) && (
            <div className="flex flex-col items-center justify-center h-64 text-center border-2 border-dashed border-gray-300 rounded-lg p-12">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-200 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-gray-400 text-lg">
                {selectedRecording ? 'Loading details...' : 'Select a recording from the list above.'}
              </p>
              <p className="text-gray-400 mt-2">
                Upload an ultrasound recording to visualize and analyze multi-view data.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Custom scrollbar styles */}
      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f3f4f6;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #d1d5db;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #9ca3af;
        }
        
        /* Slider thumb styling for webkit browsers */
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 14px;
          height: 14px;
          background: #4285f4;
          cursor: pointer;
          border-radius: 50%;
          border: 2px solid white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Slider thumb styling for Firefox */
        input[type=range]::-moz-range-thumb {
          width: 14px;
          height: 14px;
          background: #4285f4;
          cursor: pointer;
          border-radius: 50%;
          border: 2px solid white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
      `}</style>
    </div>
  );
};

export default DatasetExplorer; 