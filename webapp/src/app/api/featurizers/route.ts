import { NextResponse } from 'next/server';

export async function GET() {
  // TODO: Replace with backend fetch when available
  const featurizers = ["jbu_dinov2", "jbu_dino16"];
  return NextResponse.json(featurizers);
} 