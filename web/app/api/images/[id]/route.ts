import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { resolveImage } from "@/lib/data";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  try {
    const file = await resolveImage(id);
    if (!file) return NextResponse.json({ error: "Image not found" }, { status: 404 });
    const bytes = await fs.readFile(file);
    return new NextResponse(bytes, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "private, max-age=3600",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    return NextResponse.json({ error: "Image unavailable" }, { status: 404 });
  }
}
