import { useEffect, useState } from "react";
import { apiFetch, logout, refreshSession, setUnauthorizedHandler } from "./api.js";
import LoginScreen from "./LoginScreen.jsx";
import Workspace from "./components/Workspace.jsx";

// Auth gate. On load there is no access token in memory (it never touches
// storage), so the session is restored from the httpOnly refresh cookie.
export default function App() {
  // undefined = still checking for a session, null = logged out
  const [username, setUsername] = useState(undefined);

  useEffect(() => {
    setUnauthorizedHandler(() => setUsername(null));
    refreshSession()
      .then((ok) => (ok ? apiFetch("/auth/me").then((r) => r.json()) : null))
      .then((me) => setUsername(me ? me.username : null))
      .catch(() => setUsername(null));
  }, []);

  async function handleLogout() {
    await logout().catch(() => {});
    setUsername(null);
  }

  if (username === undefined) return null;
  if (username === null) return <LoginScreen onLoggedIn={setUsername} />;
  return <Workspace username={username} onLogout={handleLogout} />;
}
