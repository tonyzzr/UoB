import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    // Path to features directory - adjusted for project structure
    const featuresDir = path.join(process.cwd(), '..', 'configs', 'features');
    
    // Read directory contents
    const files = fs.readdirSync(featuresDir);
    
    // Filter for .toml files and extract featurizer names (remove .toml extension)
    const featurizers = files
      .filter(file => file.endsWith('.toml'))
      .map(file => file.replace('.toml', ''));
    
    return NextResponse.json(featurizers);
  } catch (error) {
    console.error('Error fetching featurizers:', error);
    // Fallback to hardcoded values in case of error
    const featurizers = ["jbu_dinov2", "jbu_dino16"];
    return NextResponse.json(featurizers);
  }
} 