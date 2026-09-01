"use client";

export default function ErrorPage({ reset }: { reset: () => void }) {
  return (
    <main className="loading-shell">
      <p>Evidence workspace could not load.</p>
      <button className="button primary" onClick={reset}>Try again</button>
    </main>
  );
}
