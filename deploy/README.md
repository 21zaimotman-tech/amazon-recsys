# Deploying ElectroPicks

The whole stack is docker-compose, so any Linux VPS (Hetzner/DigitalOcean/
Lightsail, ~$5/mo) or a container platform (Fly.io, Railway) works.

## VPS walkthrough (Caddy + automatic HTTPS)

1. Point a DNS A record for your domain (e.g. `shop.example.com`) at the VPS.
2. On the VPS: install Docker, clone the repo, copy the data the containers
   need — `artifacts/` (webapp models) and a Postgres dump or the parquet
   files + `scripts/populate_db.py`.
3. Create `.env` with real credentials (never commit it):
   ```
   POSTGRES_DB=recsys
   POSTGRES_USER=recsys
   POSTGRES_PASSWORD=<strong password>
   ```
4. Launch with the production overlay (adds Caddy, unpublishes internal
   ports, enables restart policies):
   ```bash
   DOMAIN=shop.example.com docker compose \
       -f docker-compose.yml -f docker-compose.prod.yml up -d --build
   ```
5. Populate the database once:
   ```bash
   docker compose exec api python -m scripts.populate_db   # or run locally against the VPS
   ```

Caddy fetches and renews the TLS certificate automatically. The site is at
`https://shop.example.com`, the API (Swagger) at `https://shop.example.com/api/docs`.

Note: `docker-compose.override.yml` is a local development override (it
remaps the Postgres port); compose only auto-loads it when you run plain
`docker compose up`, so the explicit `-f` files above ignore it — nothing
to change for production.

## Operational notes

- **Streamlit needs websockets** — Caddy's `reverse_proxy` handles the
  upgrade automatically; if you swap in nginx, proxy `Upgrade`/`Connection`
  headers for `/`.
- **Backups**: the only stateful pieces are the `pgdata` volume and
  `artifacts/`. `docker compose exec postgres pg_dump -U recsys recsys | gzip`
  on a cron is enough for a demo product.
- **Updating models**: replace `artifacts/` and `docker compose restart api`
  (it loads artifacts once at startup).
- **Debug mode**: append `?debug=1` to the URL for model labels, latency
  pills, the diversity slider, and the analytics dashboard.
