import { NextResponse } from 'next/server';

// Define the base URL for the Python data service
const DATA_SERVICE_URL = 'http://localhost:8000'; // Make sure this matches where the Python service is running

export async function GET(
  request: Request,
  { params }: { params: { recording_id: string } } // Uses recording_id
) {
  const recording_id = params.recording_id; // Uses recording_id
  const detailsUrl = `${DATA_SERVICE_URL}/recordings/${recording_id}/details`;

  console.log(`Next API Route: Forwarding request for ${recording_id} to ${detailsUrl}`);

  try {
    // Fetch data from the Python data service
    const response = await fetch(detailsUrl, {
      cache: 'no-store',
      headers: {
        'Accept': 'application/json',
      }
    });

    // Check if the data service responded successfully
    if (!response.ok) {
      let errorBody;
      try {
        errorBody = await response.json();
      } catch (parseError) {
        errorBody = { error: response.statusText };
      }
      console.error(`Next API Route: Error from data service (${response.status}):`, errorBody);
      return NextResponse.json(
        { error: errorBody?.detail || errorBody?.error || `Data service failed with status ${response.status}` },
        { status: response.status }
      );
    }

    const details = await response.json();
    console.log(`Next API Route: Successfully received details for ${recording_id} from data service.`);

    return NextResponse.json(details);

  } catch (error: any) {
    console.error(`Next API Route: Network error fetching from data service for ${recording_id}:`, error);
    if (error.code === 'ECONNREFUSED') {
         return NextResponse.json({ error: `Could not connect to data service at ${DATA_SERVICE_URL}. Is it running?` }, { status: 503 });
    }
    return NextResponse.json({ error: 'Internal Server Error forwarding request to data service' }, { status: 500 });
  }
} 