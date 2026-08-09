// OptiTrade Bot — service worker
// Only caches the static app shell (for fast loads / installability).
// Live data (/api/*, /webhook) is NEVER cached — always goes to network,
// so you never see stale P&L or positions.

const CACHE_NAME = "optitrade-shell-v1";
const SHELL_ASSETS = [
  "/static/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never cache live data or login/session-bearing routes.
  if (
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/webhook") ||
    url.pathname.startsWith("/login") ||
    url.pathname.startsWith("/telegram")
  ) {
    return; // let the browser handle it normally (network only)
  }

  // App shell assets: cache-first for speed.
  if (SHELL_ASSETS.some((a) => url.pathname === a)) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
    return;
  }

  // Everything else (the dashboard page itself): network-first,
  // falling back to cache only if fully offline.
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
