# Durable optimizer snapshot storage (Render + PostgreSQL)

The optimizer now stores its latest compact snapshot in PostgreSQL whenever
`DATABASE_URL` is configured. Without that variable, local development falls
back to the existing SQLite setting. SQLite on Render's ephemeral filesystem
is not durable across rebuilds/redeploys.

## Enable it on Render

1. In the Render dashboard, create a **PostgreSQL** database (choose a region
   compatible with the web service). Check the database's plan, expiry, and
   storage limits before relying on it for long-term retention.
2. Open the database's **Connect** section and copy its **Internal Database URL**
   when the web service is in the same Render region. Use the External URL only
   if the service cannot reach the internal address.
3. Open the web service for this repository → **Environment**.
4. Add an environment variable named `DATABASE_URL` and set it to the database
   connection URL. Keep the URL secret; do not commit it to GitHub or paste it
   into chat.
5. Save the environment changes and let Render redeploy the service. The
   `psycopg` driver is installed from `requirements.txt`.
6. Open the Strategy Lab, run the optimizer, then refresh the page and use
   **Restore last run**. The snapshot should survive service redeploys as long
   as the PostgreSQL database remains available.

## What is stored

- The latest compact optimizer snapshot and its phase/results/diagnostics.
- A single snapshot key is maintained to limit database growth; this is not
  yet a versioned run history.
- The first PostgreSQL read attempts a best-effort import of the existing
  SQLite snapshot if one is present and PostgreSQL has no saved snapshot.

## Notes

- This persists completed optimizer phases/results, but does **not** resume CPU
  work from the exact interrupted loop iteration.
- If PostgreSQL is configured but unavailable, reads/writes fail rather than
  silently claiming durable persistence. The response hook intentionally does
  not fail an otherwise successful optimizer response if a save fails.
- Database credentials are read only from the environment. Do not place
  credentials in source control.
