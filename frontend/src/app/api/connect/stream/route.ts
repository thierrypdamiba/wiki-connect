import { NextRequest } from "next/server";

const ADK_BACKEND_URL = process.env.ADK_BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();

  // Forward the request to the backend streaming endpoint
  const response = await fetch(`${ADK_BACKEND_URL}/connect/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic_a: body.topicA,
      topic_b: body.topicB,
      user_id: body.userId || "default",
    }),
  });

  // Return the SSE stream directly
  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
