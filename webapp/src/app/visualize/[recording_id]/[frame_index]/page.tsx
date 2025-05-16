import React from 'react';
import FeatureVisualizer from '@/components/FeatureVisualizer'; // Import the component

interface VisualizePageProps {
  params: {
    recording_id: string;
    frame_index: string;
  };
}

// This is a Server Component by default in Next.js App Router
// Define as a standard function component
function VisualizePage({ params }: VisualizePageProps) {
  // Don't destructure here
  // const { recording_id, frame_index } = params;

  // Remove the validation check that accesses params properties here
  // // Basic validation or error handling for params could go here
  // // Check if params themselves exist
  // if (!params || !params.recording_id || !params.frame_index) {
  //   return <div>Error: Missing recording ID or frame index parameters.</div>;
  // }

  // Directly pass params to the client component
  return (
    <FeatureVisualizer 
      params={params} // Pass the whole params object
      // recordingId={recording_id} 
      // frameIndex={frame_index} 
    />
  );
}

export default VisualizePage; 