import { NextRequest, NextResponse } from "next/server";

const ADK_BACKEND_URL = process.env.ADK_BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const { articleId, feedbackType, feedbackText, rating } = await req.json();

    if (!articleId || !feedbackText) {
      return NextResponse.json(
        { error: "Article ID and feedback text are required" },
        { status: 400 }
      );
    }

    // Call ADK backend
    const response = await fetch(`${ADK_BACKEND_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        article_id: articleId,
        feedback_type: feedbackType || "text_style",
        feedback_text: feedbackText,
        rating: rating || 3,
      }),
    });

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error submitting feedback:", error);
    return NextResponse.json(
      { error: "Failed to submit feedback" },
      { status: 500 }
    );
  }
}
