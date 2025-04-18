import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Environment variable for the backend service URL
const DATA_SERVICE_URL = process.env.DATA_SERVICE_URL || 'http://localhost:8000';

// Define the expected request body structure (optional but good practice)
interface CorrespondenceApiRequest {
  source_view_index: number;
  poi_normalized: [number, number];
}

export async function POST(
  request: NextRequest,
  context: { params: { recording_id: string; frame_index: string } }
) {
  try {
    // Properly handle params by destructuring them after awaiting the context object
    const { recording_id, frame_index } = context.params;

    console.log(`[API Proxy] Processing correspondence visualization request for recording: ${recording_id}, frame: ${frame_index}`);

    if (!recording_id || !frame_index) {
      return NextResponse.json({ error: 'Missing recording_id or frame_index' }, { status: 400 });
    }

    let requestBody: CorrespondenceApiRequest;
    try {
      requestBody = await request.json();
      // Basic validation
      if (requestBody.source_view_index === undefined || requestBody.source_view_index === null || 
          !requestBody.poi_normalized || requestBody.poi_normalized.length !== 2 ||
          typeof requestBody.poi_normalized[0] !== 'number' || typeof requestBody.poi_normalized[1] !== 'number') {
          throw new Error('Invalid request body structure');
      }
    } catch (error) {
      console.error('[API Proxy] Invalid request body:', error);
      return NextResponse.json({ 
        error: 'Invalid request body',
        details: error instanceof Error ? error.message : String(error)
      }, { status: 400 });
    }

    const targetUrl = `${DATA_SERVICE_URL}/recordings/${recording_id}/visualize_correspondence/${frame_index}`;

    console.log(`[API Proxy] Forwarding correspondence visualization request to: ${targetUrl}`);
    console.log(`[API Proxy] Request Body:`, requestBody);

    try {
      const backendResponse = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        cache: 'no-store', // Ensure fresh results
      });

      if (!backendResponse.ok) {
        // Forward the error response from the backend if possible
        let errorBody = { error: `Backend error: ${backendResponse.status} ${backendResponse.statusText}` };
        try {
            const backendError = await backendResponse.json();
            errorBody = { error: backendError.detail || JSON.stringify(backendError) };
        } catch (e) { /* Ignore if backend error is not JSON */ }
        console.error('[API Proxy] Backend error:', errorBody);
        return NextResponse.json(errorBody, { status: backendResponse.status });
      }

      // The response is a PNG image, so pass it through
      const imageBuffer = await backendResponse.arrayBuffer();
      
      // Create a new response with the image data and correct content type
      return new NextResponse(imageBuffer, {
        status: 200,
        headers: {
          'Content-Type': 'image/png',
          'Cache-Control': 'public, max-age=60', // Cache for 1 minute
        },
      });

    } catch (error) {
      console.error('[API Proxy] Error fetching from data service:', error);
      return NextResponse.json({ 
        error: 'Failed to fetch from data service',
        details: error instanceof Error ? error.message : String(error)
      }, { status: 502 }); // Bad Gateway
    }
  } catch (outerError) {
    console.error('[API Proxy] Unexpected error in correspondence visualization route handler:', outerError);
    return NextResponse.json({
      error: 'Unexpected error in API route handler',
      details: outerError instanceof Error ? outerError.message : String(outerError)
    }, { status: 500 });
  }
} 