'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link'; // For back button

interface FeatureVisualizerProps {
  recordingId: string;
  frameIndex: string; // Initially passed as string from route param
}

// Interface for the expected API response
interface FeatureVizData {
    recording_id: string;
    frame_index: number;
    input_image_uris: string[]; // List of 16 Data URIs
    pca_image_uris: string[];   // List of 16 Data URIs
    view_labels: string[];      // List of 16 labels (e.g., "LF 0")
}

const FeatureVisualizer: React.FC<FeatureVisualizerProps> = ({ recordingId, frameIndex }) => {
  const [data, setData] = useState<FeatureVizData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Construct the API URL to the Next.js proxy route
        const apiUrl = `/api/recordings/${recordingId}/visualize_features/${frameIndex}`;
        console.log(`Fetching feature visualization data from: ${apiUrl}`);
        const response = await fetch(apiUrl);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result: FeatureVizData = await response.json();
        setData(result);
        console.log("Successfully fetched and parsed feature visualization data.");
      } catch (err: any) {
        console.error("Failed to fetch feature visualization data:", err);
        setError(err.message || "An unknown error occurred while fetching data.");
        setData(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [recordingId, frameIndex]); // Re-fetch if props change

  // Helper to split data into LF and HF rows
  const splitData = (uris: string[], labels: string[]) => {
      const numViews = labels.length;
      const halfPoint = Math.ceil(numViews / 2); // Assumes LF/HF split
      const lfData = uris.slice(0, halfPoint).map((uri, i) => ({ uri, label: labels[i] }));
      const hfData = uris.slice(halfPoint).map((uri, i) => ({ uri, label: labels[halfPoint + i] }));
      return { lfData, hfData };
  };

  const inputSplit = data ? splitData(data.input_image_uris, data.view_labels) : { lfData: [], hfData: [] };
  const pcaSplit = data ? splitData(data.pca_image_uris, data.view_labels) : { lfData: [], hfData: [] };

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
            <p>Recording: <span className="font-semibold">{recordingId}</span></p>
            <p>Frame: <span className="font-semibold">{frameIndex}</span></p>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center space-x-2 text-gray-600 py-20">
              <svg className="animate-spin h-8 w-8" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-lg">Loading visualization data...</p>
            </div>
          )}

          {error && (
            <div className="border-2 border-dashed border-red-300 bg-red-50 rounded-lg p-12 text-center">
              <p className="text-red-600 font-semibold text-lg">Error Loading Features:</p>
              <p className="text-red-500 mt-2">{error}</p>
            </div>
          )}

          {!isLoading && !error && data && (
            <div className="space-y-6">
              {/* 4x8 Grid */} 
              <div className="grid grid-cols-8 gap-2">
                  {/* Row 1: Input LF */}
                  {inputSplit.lfData.map(({ uri, label }) => (
                      <div key={label + '-input'} className="text-center">
                          <p className="text-xs font-medium text-gray-600 mb-1">Input {label}</p>
                          <div className="aspect-w-1 aspect-h-1 border border-gray-200 rounded overflow-hidden bg-white flex items-center justify-center">
                              <img src={uri} alt={`Input ${label}`} className="max-w-full max-h-full object-contain" />
                          </div>
                      </div>
                  ))}

                  {/* Row 2: PCA LF */}
                  {pcaSplit.lfData.map(({ uri, label }) => (
                      <div key={label + '-pca'} className="text-center">
                          <p className="text-xs font-medium text-gray-600 mb-1">PCA {label}</p>
                           <div className="aspect-w-1 aspect-h-1 border border-gray-200 rounded overflow-hidden bg-white flex items-center justify-center">
                              <img src={uri} alt={`PCA ${label}`} className="max-w-full max-h-full object-contain" />
                          </div>
                      </div>
                  ))}
                  
                  {/* Row 3: Input HF */}
                   {inputSplit.hfData.map(({ uri, label }) => (
                      <div key={label + '-input'} className="text-center">
                          <p className="text-xs font-medium text-gray-600 mb-1">Input {label}</p>
                           <div className="aspect-w-1 aspect-h-1 border border-gray-200 rounded overflow-hidden bg-white flex items-center justify-center">
                              <img src={uri} alt={`Input ${label}`} className="max-w-full max-h-full object-contain" />
                          </div>
                      </div>
                  ))}
                  
                  {/* Row 4: PCA HF */}
                  {pcaSplit.hfData.map(({ uri, label }) => (
                      <div key={label + '-pca'} className="text-center">
                          <p className="text-xs font-medium text-gray-600 mb-1">PCA {label}</p>
                          <div className="aspect-w-1 aspect-h-1 border border-gray-200 rounded overflow-hidden bg-white flex items-center justify-center">
                              <img src={uri} alt={`PCA ${label}`} className="max-w-full max-h-full object-contain" />
                          </div>
                      </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default FeatureVisualizer; 