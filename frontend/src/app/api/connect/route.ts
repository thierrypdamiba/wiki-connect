import { NextRequest, NextResponse } from "next/server";

const ADK_BACKEND_URL = process.env.ADK_BACKEND_URL || "http://localhost:9843";

export async function POST(req: NextRequest) {
  try {
    const { topicA, topicB } = await req.json();

    if (!topicA || !topicB) {
      return NextResponse.json(
        { error: "Both topics are required" },
        { status: 400 }
      );
    }

    // Call ADK backend
    const response = await fetch(`${ADK_BACKEND_URL}/connect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic_a: topicA, topic_b: topicB }),
    });

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error connecting topics:", error);
    return NextResponse.json(
      { error: "Failed to connect topics" },
      { status: 500 }
    );
  }
}
