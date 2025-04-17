import { NextRequest, NextResponse } from 'next/server';

// Assume the Python data service is running on port 8000
const DATA_SERVICE_URL = process.env.DATA_SERVICE_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest, // Request object is not used here but required by signature
  { params }: { params: { recording_id: string; frame_index: string } }
) {
  const { recording_id, frame_index } = params;

  if (!recording_id || !frame_index) {
    return NextResponse.json({ error: 'Missing recording ID or frame index' }, { status: 400 });
  }

  console.log(`[Next API Proxy] Forwarding feature visualization request for ${recording_id}, frame ${frame_index} to data service...`);

  try {
    const targetUrl = `${DATA_SERVICE_URL}/recordings/${recording_id}/visualize_features/${frame_index}`;
    console.log(`[Next API Proxy] Target URL: ${targetUrl}`);
    
    const response = await fetch(targetUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      cache: 'no-store', // Ensure fresh data is fetched from the service
    });

    // Check if the data service responded successfully
    if (!response.ok) {
      // Try to parse error from data service, otherwise use status text
      const errorBody = await response.json().catch(() => ({ detail: response.statusText }));
      console.error(`[Next API Proxy] Error from data service: ${response.status}`, errorBody);
      return NextResponse.json(
        { error: `Data service error: ${errorBody.detail || response.statusText}` }, 
        { status: response.status }
      );
    }

    // Parse the successful JSON response from the data service
    const data = await response.json();
    console.log(`[Next API Proxy] Successfully received data from service for ${recording_id}, frame ${frame_index}.`);
    
    // Return the data to the client component
    return NextResponse.json(data);

  } catch (error: any) {
    console.error('[Next API Proxy] Internal error proxying request:', error);
    return NextResponse.json(
      { error: `Internal proxy error: ${error.message || 'Unknown error'}` }, 
      { status: 500 }
    );
  }
} 