import { NextRequest, NextResponse } from 'next/server';

// Assume the Python data service is running on port 8000
const DATA_SERVICE_URL = process.env.DATA_SERVICE_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest, // Request object is not used here but required by signature
  context: { params: { recording_id: string; frame_index: string } }
) {
  try {
    // Properly handle params by destructuring them after awaiting the context object
    const { recording_id, frame_index } = context.params;

    if (!recording_id || !frame_index) {
      return NextResponse.json({ error: 'Missing recording ID or frame index' }, { status: 400 });
    }

    console.log(`[Next API Proxy] Forwarding feature visualization request for ${recording_id}, frame ${frame_index} to data service...`);

    try {
      const targetUrl = `${DATA_SERVICE_URL}/recordings/${recording_id}/visualize_features/${frame_index}`;
      console.log(`[Next API Proxy] Target URL: ${targetUrl}`);
      
      const response = await fetch(targetUrl, {
        method: 'GET',
        // Remove Accept header or change it if necessary, but backend returns image now
        // headers: {
        //   'Accept': 'application/json',
        // },
        cache: 'no-store', // Ensure fresh data is fetched from the service
      });

      // Check if the data service responded successfully (status 200-299)
      if (!response.ok) {
        // Data service returned an error. Forward the status and try to get error text.
        const errorText = await response.text().catch(() => response.statusText);
        console.error(`[Next API Proxy] Error from data service: ${response.status} ${errorText}`);
        return new Response(
          `Data service error: ${errorText}`,
          { status: response.status }
        );
      }

      // --- FIX: Handle Image Response --- 
      // Get the response body as a Blob (binary large object)
      const imageBlob = await response.blob();
      console.log(`[Next API Proxy] Successfully received image blob from service for ${recording_id}, frame ${frame_index}. Type: ${imageBlob.type}`);
      
      // Return the image blob directly, setting the correct Content-Type
      // The browser calling this API route will know how to handle the image.
      return new NextResponse(imageBlob, {
        status: 200,
        headers: {
          'Content-Type': imageBlob.type || 'image/png', // Use the blob's type or default to PNG
          'Cache-Control': 'public, max-age=300', // Cache for 5 minutes
        },
      });
      // ----------------------------------

    } catch (error: any) {
      console.error('[Next API Proxy] Internal error proxying request:', error);
      // Use Response for consistency, even for internal errors
      return new Response(
        `Internal proxy error: ${error.message || 'Unknown error'}`,
        { status: 500 }
      );
    }
  } catch (outerError: any) {
    console.error('[Next API Proxy] Unexpected error in route handler:', outerError);
    return new Response(
      `Route handler error: ${outerError.message || 'Unknown error'}`,
      { status: 500 }
    );
  }
} 