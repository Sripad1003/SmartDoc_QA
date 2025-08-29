export function getApiBase() {
  // Use NEXT_PUBLIC_API_BASE_URL if available; default to local FastAPI
  const base = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:8000"
  return base.replace(/\/+$/, "")
}
