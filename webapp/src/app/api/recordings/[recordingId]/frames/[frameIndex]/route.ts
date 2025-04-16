import { NextResponse } from 'next/server';
import { type NextRequest } from 'next/server'

const DATA_SERVICE_URL = 'http://localhost:8000'; // Ensure this matches the Python service

export async function GET(
  request: NextRequest, // Use NextRequest to easily access searchParams
  { params }: { params: { recordingId: string; frameIndex: string } }
) {
  const recordingId = params.recordingId;
  const frameIndex = params.frameIndex;

  // Get query parameters (freq, view) from the original request
  const searchParams = request.nextUrl.searchParams;
  const freq = searchParams.get('freq');
  const view = searchParams.get('view');

  if (!freq || !view) {
    return NextResponse.json({ error: 'Missing required query parameters: freq, view' }, { status: 400 });
  }

  const pythonServiceUrl = `${DATA_SERVICE_URL}/recordings/${recordingId}/frames/${frameIndex}?freq=${encodeURIComponent(freq)}&view=${encodeURIComponent(view)}`;

  console.log(`Next Frame API: Forwarding request to ${pythonServiceUrl}`);

  try {
    // Fetch the image from the Python data service
    const response = await fetch(pythonServiceUrl, {
      cache: 'no-store', // Don't cache images between services
    });

    // Check if the data service responded successfully (status 2xx)
    if (!response.ok) {
      console.error(`Next Frame API: Error from data service (${response.status})`);
      // Return an appropriate error response, perhaps just the status?
      // Avoid returning potentially large error bodies from the service here.
      return new NextResponse(response.body, { status: response.status, headers: {'Content-Type': 'application/json'} });
      // Or a generic error:
      // return NextResponse.json({ error: `Data service failed with status ${response.status}` }, { status: response.status });
    }

    // Check content type - MUST be an image
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.startsWith('image/')) {
         console.error(`Next Frame API: Data service did not return an image. Content-Type: ${contentType}`);
         return NextResponse.json({ error: 'Data service did not return an image' }, { status: 502 }); // Bad Gateway
    }

    console.log(`Next Frame API: Successfully received image from data service. Content-Type: ${contentType}`);

    // Stream the image response back to the client
    // Ensure headers like Content-Type and potentially Content-Length are forwarded
    const headers = new Headers();
    headers.set('Content-Type', contentType);
    // Add other headers if needed, e.g., Content-Length if available from response.headers

    // Use NextResponse to stream the body directly
    return new NextResponse(response.body, {
        status: 200,
        headers: headers,
    });

  } catch (error: any) {
    // Handle network errors connecting to the data service
    console.error(`Next Frame API: Network error fetching from data service:`, error);
     if (error.code === 'ECONNREFUSED') {
         return NextResponse.json({ error: `Could not connect to data service at ${DATA_SERVICE_URL}. Is it running?` }, { status: 503 }); // Service Unavailable
    }
    return NextResponse.json({ error: 'Internal Server Error forwarding frame request' }, { status: 500 });
  }
} 