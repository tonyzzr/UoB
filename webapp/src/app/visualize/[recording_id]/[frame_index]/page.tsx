import React from 'react';
import FeatureVisualizer from '@/components/FeatureVisualizer'; // Import the component

interface VisualizePageProps {
  params: {
    recording_id: string;
    frame_index: string;
  };
}

// This is a Server Component by default in Next.js App Router
const VisualizePage: React.FC<VisualizePageProps> = ({ params }) => {
  const { recording_id, frame_index } = params;

  // Basic validation or error handling for params could go here
  if (!recording_id || !frame_index) {
    return <div>Error: Missing recording ID or frame index.</div>;
  }

  return (
    <FeatureVisualizer 
      recordingId={recording_id} 
      frameIndex={frame_index} 
    />
  );
};

export default VisualizePage; 