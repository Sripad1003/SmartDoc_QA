export function getApiBase() {
  // Prefer server-side env; fallback to NEXT_PUBLIC for flexibility.
  const raw =
    process.env.API_BASE_URL?.trim() || process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:8000"
  return raw.replace(/\/+$/, "")
}
