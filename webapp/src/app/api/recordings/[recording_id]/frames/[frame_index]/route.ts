import { NextResponse } from 'next/server';
import { type NextRequest } from 'next/server'

const DATA_SERVICE_URL = 'http://localhost:8000'; // Ensure this matches the Python service

export async function GET(
  request: NextRequest, // Use NextRequest to easily access searchParams
  context: { params: { recording_id: string; frame_index: string } } // Uses recording_id and frame_index
) {
  try {
    // Properly handle params by destructuring them after awaiting the context object
    const { recording_id, frame_index } = context.params;
    
    console.log(`[Next Frames API] Processing frame request for recording: ${recording_id}, frame: ${frame_index}`);

    if (!recording_id || !frame_index) {
      return NextResponse.json({ error: 'Missing required parameters: recording_id, frame_index' }, { status: 400 });
    }

    // Get query parameters (freq, view) from the original request
    const searchParams = request.nextUrl.searchParams;
    const freq = searchParams.get('freq');
    const view = searchParams.get('view');
    
    console.log(`[Next Frames API] Query params: freq=${freq}, view=${view}`);

    if (!freq || !view) {
      console.error(`[Next Frames API] Missing required parameters: freq=${freq}, view=${view}`);
      return NextResponse.json({ error: 'Missing required query parameters: freq, view' }, { status: 400 });
    }

    const pythonServiceUrl = `${DATA_SERVICE_URL}/recordings/${recording_id}/frames/${frame_index}?freq=${encodeURIComponent(freq)}&view=${encodeURIComponent(view)}`;

    console.log(`[Next Frames API] Forwarding request to ${pythonServiceUrl}`);

    try {
      // Fetch the image from the Python data service
      const response = await fetch(pythonServiceUrl, {
        cache: 'no-store', // Don't cache images between services
      });

      // Check if the data service responded successfully (status 2xx)
      if (!response.ok) {
        console.error(`[Next Frames API] Error from data service (${response.status}): ${await response.text()}`);
        return new NextResponse(response.body, { status: response.status, headers: {'Content-Type': 'application/json'} });
      }

      // Check content type - MUST be an image
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.startsWith('image/')) {
          console.error(`[Next Frames API] Data service did not return an image. Content-Type: ${contentType}`);
          return NextResponse.json({ error: 'Data service did not return an image' }, { status: 502 }); // Bad Gateway
      }

      console.log(`[Next Frames API] Successfully received image from data service. Content-Type: ${contentType}`);

      // Stream the image response back to the client
      const headers = new Headers();
      headers.set('Content-Type', contentType);
      // Set caching headers
      headers.set('Cache-Control', 'public, max-age=300'); // Cache for 5 minutes

      return new NextResponse(response.body, {
          status: 200,
          headers: headers,
      });

    } catch (error: any) {
      console.error(`[Next Frames API] Network error fetching from data service:`, error);
      if (error.code === 'ECONNREFUSED') {
          return NextResponse.json({ 
            error: `Could not connect to data service at ${DATA_SERVICE_URL}. Is it running?`,
            details: error.message
          }, { status: 503 }); // Service Unavailable
      }
      return NextResponse.json({ 
        error: 'Internal Server Error forwarding frame request',
        details: error.message
      }, { status: 500 });
    }
  } catch (outerError: any) {
    console.error(`[Next Frames API] Unexpected error in route handler:`, outerError);
    return NextResponse.json({ 
      error: 'Unexpected error in API route handler',
      details: outerError.message
    }, { status: 500 });
  }
} 