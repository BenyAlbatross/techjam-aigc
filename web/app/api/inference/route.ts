import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 300;

const execute = promisify(execFile);
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const ALLOWED_MODELS = new Set(["ateeqq_siglip", "community_forensics", "univfd"]);
const ALLOWED_TYPES = new Map([
  ["image/jpeg", ".jpg"],
  ["image/png", ".png"],
  ["image/webp", ".webp"],
]);

export async function POST(request: Request) {
  const form = await request.formData();
  const upload = form.get("image");
  const model = String(form.get("model") ?? "ateeqq_siglip");
  if (!(upload instanceof File)) {
    return NextResponse.json({ error: "Choose an image first." }, { status: 400 });
  }
  if (!ALLOWED_MODELS.has(model)) {
    return NextResponse.json({ error: "Model is not enabled for ad hoc testing." }, { status: 400 });
  }
  const suffix = ALLOWED_TYPES.get(upload.type);
  if (!suffix || upload.size > 10 * 1024 * 1024) {
    return NextResponse.json({ error: "Use PNG, JPEG, or WebP up to 10 MB." }, { status: 400 });
  }

  const session = randomUUID();
  const sessionRoot = path.join(PROJECT_ROOT, "work", "ad-hoc", session);
  const input = path.join(sessionRoot, "input");
  const output = path.join(sessionRoot, "predictions.json");
  await fs.mkdir(input, { recursive: true });
  await fs.writeFile(path.join(input, `upload${suffix}`), Buffer.from(await upload.arrayBuffer()));

  try {
    await execute(
      "pixi",
      [
        "run", "--platform", "linux-aarch64-cuda", "infer", "--", "--model", model,
        "--input", input, "--output", output, "--device", "cuda:0", "--cache", "work/hf-cache",
      ],
      { cwd: PROJECT_ROOT, timeout: 280_000, env: { ...process.env, HF_HUB_OFFLINE: "1" } },
    );
    const rows = JSON.parse(await fs.readFile(output, "utf8")) as Array<{ image_path: string; pred: number }>;
    return NextResponse.json({ session, model, probabilityAi: rows[0]?.pred, status: "complete" });
  } catch {
    await fs.rm(sessionRoot, { recursive: true, force: true });
    return NextResponse.json(
      { error: "Local CUDA model is unavailable. Gallery evidence remains usable." },
      { status: 503 },
    );
  }
}

export async function DELETE(request: Request) {
  const session = new URL(request.url).searchParams.get("session") ?? "";
  if (!/^[0-9a-f-]{36}$/.test(session)) {
    return NextResponse.json({ error: "Invalid session." }, { status: 400 });
  }
  await fs.rm(path.join(PROJECT_ROOT, "work", "ad-hoc", session), { recursive: true, force: true });
  return NextResponse.json({ cleared: true });
}
