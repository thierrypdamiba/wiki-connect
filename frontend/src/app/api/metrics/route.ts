import { NextResponse } from "next/server";
import { getMetrics, getAllCollectionsInfo, cleanupExpiredCache } from "@/lib/qdrant";

export async function GET() {
  try {
    const [metrics, collections] = await Promise.all([
      Promise.resolve(getMetrics()),
      getAllCollectionsInfo(),
    ]);

    return NextResponse.json({
      ...metrics,
      collections,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return NextResponse.json(
      { error: "Failed to fetch metrics" },
      { status: 500 }
    );
  }
}

export async function POST() {
  try {
    const deletedCount = await cleanupExpiredCache();

    return NextResponse.json({
      message: "Cache cleanup completed",
      deletedCount,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error("Error during cache cleanup:", error);
    return NextResponse.json(
      { error: "Failed to cleanup cache" },
      { status: 500 }
    );
  }
}
