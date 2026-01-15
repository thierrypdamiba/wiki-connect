import { NextRequest, NextResponse } from "next/server";
import { buildImagePrompt } from "@/lib/gemini";

export async function POST(req: NextRequest) {
  try {
    const { topicA, topicB, connectionSummary } = await req.json();

    if (!topicA || !topicB) {
      return NextResponse.json(
        { error: "Both topics are required" },
        { status: 400 }
      );
    }

    // For now, return null - image generation requires special Gemini models
    // that may not be available in all regions/accounts
    // The frontend handles this gracefully
    return NextResponse.json({
      imageId: null,
      imageUrl: null,
      prompt: buildImagePrompt(topicA, topicB, connectionSummary || `Connection between ${topicA} and ${topicB}`),
      message: "Image generation not available in serverless mode",
    });
  } catch (error) {
    console.error("Error generating image:", error);
    return NextResponse.json(
      { error: "Failed to generate image" },
      { status: 500 }
    );
  }
}
