// Backend URL. "/api" goes through a same-origin proxy (Vite's dev server,
// or nginx in Docker), which keeps the refresh-token cookie first-party.
// Set VITE_API_BASE to call the API directly instead (it must then be
// listed in the backend's CORS_ORIGINS).
export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

// The access token lives only in memory -- never in localStorage, where
// any injected script could read it. A page reload loses it, and
// refreshSession() gets a new one from the httpOnly refresh cookie, which
// JavaScript can't read at all.
let accessToken = null;

// Called when the session can't be renewed, so the app can drop back to
// the login screen.
let onUnauthorized = () => {};
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

export class ApiError extends Error {}

// Routes where a 401 means "wrong credentials / no session", not "access
// token expired" -- retrying them after a refresh makes no sense.
const NO_REFRESH = new Set(["/auth/token", "/auth/refresh", "/auth/register", "/auth/logout"]);

let refreshing = null;

// Swaps the refresh cookie for a new access token. Concurrent callers
// share one request: the cookie rotates on every use, so two parallel
// refreshes would make the second look like a stolen-token replay.
export function refreshSession() {
  if (!refreshing) {
    refreshing = (async () => {
      try {
        const response = await fetch(`${API_BASE}/auth/refresh`, { method: "POST", credentials: "include" });
        if (!response.ok) {
          accessToken = null;
          return false;
        }
        accessToken = (await response.json()).access_token;
        return true;
      } catch {
        return false;
      }
    })().finally(() => {
      refreshing = null;
    });
  }
  return refreshing;
}

// fetch() against the backend with the bearer token attached. On a 401 it
// renews the access token once and retries; if that fails the session is
// over. Throws ApiError carrying the backend's "detail" on any non-2xx.
export async function apiFetch(path, options = {}, retried = false) {
  const headers = { ...(options.headers || {}) };
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers, credentials: "include" });

  if (response.status === 401 && !NO_REFRESH.has(path)) {
    if (!retried && (await refreshSession())) return apiFetch(path, options, true);
    accessToken = null;
    onUnauthorized();
    throw new ApiError("Your session has expired — please log in again.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => d.msg).join("; ") // FastAPI validation errors
      : body.detail;
    throw new ApiError(detail || `Request failed (${response.status})`);
  }
  return response;
}

export async function login(username, password) {
  // OAuth2 password flow: form-encoded, not JSON. The response also sets
  // the refresh cookie.
  const response = await apiFetch("/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  accessToken = (await response.json()).access_token;
}

export async function logout() {
  try {
    await apiFetch("/auth/logout", { method: "POST" });
  } finally {
    accessToken = null;
  }
}

export async function register(username, password) {
  await apiFetch("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
}

// A plain <a href> can't send an Authorization header, so the PDF is
// fetched with the token and handed to the browser as a blob download.
export async function downloadFile(path, filename) {
  const response = await apiFetch(path);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
