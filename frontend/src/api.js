// Backend URL — change this if you run FastAPI on a different port/host.
export const API_BASE = "http://127.0.0.1:8000";

const TOKEN_KEY = "researchlens_token";

// The token lives in localStorage so a page refresh doesn't log you out.
// That's fine for a local tool; for a public deployment, an httpOnly
// cookie set by the backend is safer against XSS.
export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage blocked (private window etc.) -- the session just won't survive a refresh.
  }
}

// Called on any 401 so the app can drop back to the login screen.
let onUnauthorized = () => {};
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

export class ApiError extends Error {}

// fetch() against the backend with the bearer token attached. Throws
// ApiError carrying the backend's "detail" message on any non-2xx.
export async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (response.status === 401 && token) {
    setToken(null);
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
  // OAuth2 password flow: form-encoded, not JSON.
  const response = await apiFetch("/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  const { access_token } = await response.json();
  setToken(access_token);
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
