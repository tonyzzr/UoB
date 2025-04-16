import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Define the path to the processed data directory relative to the project root
// Assumes the webapp is run from the project root or can access it.
// Adjust if the execution context is different.
const PROCESSED_DATA_DIR = path.resolve(process.cwd(), '..', 'data', 'processed');
// Note: process.cwd() in Next.js API routes usually points to the *project root* (where package.json is),
// so going up one level ('..') gets us to the UoB directory root containing 'data'.

export async function GET(request: Request) {
  console.log(`API Route: Attempting to list recordings in: ${PROCESSED_DATA_DIR}`);

  try {
    // Read the directory contents
    const entries = await fs.promises.readdir(PROCESSED_DATA_DIR, { withFileTypes: true });

    // Filter for directories
    const recordingDirs = entries
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name);
      
    console.log(`API Route: Found recordings: ${recordingDirs.join(', ')}`);

    // Return the list of directory names as JSON
    return NextResponse.json({ recordings: recordingDirs });

  } catch (error: any) {
    console.error(`API Route Error: Failed to list recordings in ${PROCESSED_DATA_DIR}:`, error);
    // Handle specific errors like directory not found
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: `Processed data directory not found at ${PROCESSED_DATA_DIR}` }, { status: 404 });
    }
    // Generic error response
    return NextResponse.json({ error: 'Internal Server Error reading recordings' }, { status: 500 });
  }
} 