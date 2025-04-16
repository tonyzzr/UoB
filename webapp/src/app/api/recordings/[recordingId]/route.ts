import { NextResponse } from 'next/server';

// Define the base URL for the Python data service
const DATA_SERVICE_URL = 'http://localhost:8000'; // Make sure this matches where the Python service is running

export async function GET(
  request: Request,
  { params }: { params: { recordingId: string } }
) {
  const recordingId = params.recordingId;
  const detailsUrl = `${DATA_SERVICE_URL}/recordings/${recordingId}/details`;

  console.log(`Next API Route: Forwarding request for ${recordingId} to ${detailsUrl}`);

  try {
    // Fetch data from the Python data service
    const response = await fetch(detailsUrl, {
      cache: 'no-store', // Avoid caching responses between the services if data might change
      headers: {
        'Accept': 'application/json',
      }
    });

    // Check if the data service responded successfully
    if (!response.ok) {
      // Attempt to parse error details from the data service response
      let errorBody;
      try {
        errorBody = await response.json();
      } catch (parseError) {
        // If parsing fails, use the status text
        errorBody = { error: response.statusText };
      }
      console.error(`Next API Route: Error from data service (${response.status}):`, errorBody);
      // Forward the status code and error message from the data service
      return NextResponse.json(
        { error: errorBody?.detail || errorBody?.error || `Data service failed with status ${response.status}` },
        { status: response.status }
      );
    }

    // Parse the successful JSON response from the data service
    const details = await response.json();
    console.log(`Next API Route: Successfully received details for ${recordingId} from data service.`);

    // Return the details fetched from the Python service
    return NextResponse.json(details);

  } catch (error: any) {
    // Handle network errors connecting to the data service
    console.error(`Next API Route: Network error fetching from data service for ${recordingId}:`, error);
    if (error.code === 'ECONNREFUSED') {
         return NextResponse.json({ error: `Could not connect to data service at ${DATA_SERVICE_URL}. Is it running?` }, { status: 503 }); // Service Unavailable
    }
    return NextResponse.json({ error: 'Internal Server Error forwarding request to data service' }, { status: 500 });
  }
} 