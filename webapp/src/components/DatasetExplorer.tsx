'use client'; // Required for components using hooks like useState, useEffect

import React, { useState, useEffect } from 'react';

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

const DatasetExplorer: React.FC = () => {
  const [recordings, setRecordings] = useState<string[]>([]);
  const [selectedRecording, setSelectedRecording] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // State for recording details
  const [recordingDetails, setRecordingDetails] = useState<RecordingDetailsResponse | null>(null);
  const [isDetailsLoading, setIsDetailsLoading] = useState<boolean>(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);

  // State for frame selection
  const [selectedFrame, setSelectedFrame] = useState<number>(0);
  const [selectedFreq, setSelectedFreq] = useState<string | null>(null);
  const [selectedSpatialView, setSelectedSpatialView] = useState<number>(0);

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
      setSelectedFrame(0);
      setSelectedFreq(null);
      setSelectedSpatialView(0);
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
        if (data?.available_freq_views && data.available_freq_views.length > 0) {
          setSelectedFreq(data.available_freq_views[0]); // Default to first freq
        }
        setSelectedFrame(0); // Reset frame to 0
        setSelectedSpatialView(0); // Reset view to 0
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
  const frameCount = typeof recordingDetails?.frame_count === 'number' ? recordingDetails.frame_count : 0;
  const spatialViewCount = typeof recordingDetails?.num_spatial_views === 'number' ? recordingDetails.num_spatial_views : 0;
  const freqViews = recordingDetails?.available_freq_views || [];

  // --- Construct Image URL --- 
  // Construct the URL only when all necessary parameters are available
  const frameImageUrl = selectedRecording && selectedFreq 
    ? `/api/recordings/${selectedRecording}/frames/${selectedFrame}?freq=${selectedFreq}&view=${selectedSpatialView}`
    : null;

  return (
    <div className="p-4 border rounded-lg shadow-md bg-white">
      <h2 className="text-xl font-semibold mb-3 text-gray-700">Processed Recordings</h2>
      {isLoading && <p className="text-gray-500">Loading recordings...</p>}
      {error && <p className="text-red-600">Error: {error}</p>}
      {!isLoading && !error && (
        recordings.length > 0 ? (
          <ul className="list-disc pl-5 space-y-1">
            {recordings.map((rec) => (
              <li
                key={rec}
                className={`
                  p-1 rounded          // Re-add padding and rounding
                  text-gray-800
                  cursor-pointer
                  // Use slightly different hover/selected for clarity
                  ${selectedRecording === rec 
                    ? 'bg-blue-200 font-semibold' 
                    : 'hover:bg-gray-100'}
                `}
                onClick={() => setSelectedRecording(rec)} // Re-add the onClick handler
              >
                {rec}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500">No processed recordings found.</p>
        )
      )}
      {selectedRecording && (
        <div className="mt-4 p-4 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-600 mb-2">Details for: {selectedRecording}</h3>
          {isDetailsLoading && <p className="text-gray-500">Loading details...</p>}
          {detailsError && <p className="text-red-600">Error loading details: {detailsError}</p>}
          {recordingDetails && !isDetailsLoading && !detailsError && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Metadata Display */}
              <div>
                <p className="text-sm text-gray-600">Path: {recordingDetails.pkl_path || 'N/A'}</p>
                <p className="text-gray-800"><span className="font-semibold">Frames:</span> {frameCount}</p>
                <p className="text-gray-800">
                  <span className="font-semibold">Freq Views:</span> {freqViews.join(', ') || 'N/A'}
                </p>
                <p className="text-gray-800">
                    <span className="font-semibold">Spatial Views (per Freq):</span> {spatialViewCount}
                </p>
                
                {/* --- Frame Selection Controls --- */}
                <div className="mt-4 space-y-2">
                   {/* Frequency Selector */} 
                   <div>
                       <label htmlFor="freqSelect" className="block text-sm font-medium text-gray-700 mr-2">Frequency:</label>
                       <select 
                          id="freqSelect"
                          value={selectedFreq ?? ''} 
                          onChange={(e) => setSelectedFreq(e.target.value)}
                          disabled={freqViews.length === 0}
                          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                       >
                          {freqViews.map(f => <option key={f} value={f}>{f}</option>)}
                       </select>
                   </div>
                   
                   {/* Spatial View Selector */} 
                   <div>
                       <label htmlFor="spatialViewSelect" className="block text-sm font-medium text-gray-700 mr-2">Spatial View:</label>
                       <select 
                          id="spatialViewSelect"
                          value={selectedSpatialView} 
                          onChange={(e) => setSelectedSpatialView(Number(e.target.value))}
                          disabled={spatialViewCount === 0}
                          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                       >
                          {[...Array(spatialViewCount).keys()].map(i => <option key={i} value={i}>{i}</option>)}
                       </select>
                   </div>

                   {/* Frame Selector */} 
                   <div>
                        <label htmlFor="frameInput" className="block text-sm font-medium text-gray-700">Frame (0-{frameCount > 0 ? frameCount - 1 : 0}):</label>
                        <input 
                            type="number" 
                            id="frameInput"
                            value={selectedFrame}
                            min="0"
                            max={frameCount > 0 ? frameCount - 1 : 0}
                            onChange={(e) => {
                                const val = Math.max(0, Math.min(Number(e.target.value), frameCount > 0 ? frameCount - 1 : 0));
                                setSelectedFrame(isNaN(val) ? 0 : val);
                            }}
                            disabled={frameCount === 0}
                            className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        />
                        {/* Optional: Add slider later */} 
                   </div>
                </div>
              </div>

              {/* Image Display */}
              <div className="flex items-center justify-center border border-gray-200 p-2 min-h-[200px]">
                {frameImageUrl ? (
                  <img 
                    key={frameImageUrl} /* Add key to force reload on src change */
                    src={frameImageUrl} 
                    alt={`Frame ${selectedFrame}, Freq ${selectedFreq}, View ${selectedSpatialView}`} 
                    className="max-w-full max-h-full object-contain"
                    onError={(e) => e.currentTarget.src = ''} // Handle image load errors
                    />
                ) : (
                  <p className="text-gray-400">Select parameters to load image</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DatasetExplorer; 