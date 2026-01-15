import { NextRequest, NextResponse } from "next/server";
import { storeFeedback } from "@/lib/qdrant";

export async function POST(req: NextRequest) {
  try {
    const { articleId, feedbackType, feedbackText, rating, userId } = await req.json();

    if (!articleId || !feedbackText) {
      return NextResponse.json(
        { error: "Article ID and feedback text are required" },
        { status: 400 }
      );
    }

    const feedback = await storeFeedback(
      userId || "default",
      articleId,
      feedbackText,
      feedbackType || "text_style",
      rating || 3
    );

    return NextResponse.json({
      feedbackId: feedback.feedback_id,
      message: "Feedback saved successfully",
    });
  } catch (error) {
    console.error("Error submitting feedback:", error);
    return NextResponse.json(
      { error: "Failed to submit feedback" },
      { status: 500 }
    );
  }
}
