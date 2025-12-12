import { NextRequest, NextResponse } from "next/server";

const ADK_BACKEND_URL = process.env.ADK_BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const { topicA, topicB, connectionSummary } = await req.json();

    if (!topicA || !topicB) {
      return NextResponse.json(
        { error: "Both topics are required" },
        { status: 400 }
      );
    }

    // Call ADK backend
    const response = await fetch(`${ADK_BACKEND_URL}/generate-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic_a: topicA,
        topic_b: topicB,
        connection_summary: connectionSummary || `Connection between ${topicA} and ${topicB}`,
      }),
    });

    const data = await response.json();

    // Convert relative URL to absolute
    if (data.imageUrl) {
      data.imageUrl = `${ADK_BACKEND_URL}${data.imageUrl}`;
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error generating image:", error);
    return NextResponse.json(
      { error: "Failed to generate image" },
      { status: 500 }
    );
  }
}
