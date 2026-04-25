# Monitoring

A scaffolded chapkit service exposes `/metrics` (Prometheus format) by default — `chapkit init` calls `.with_monitoring()` on the builder. This guide shows how to scrape that endpoint with Prometheus and surface it in Grafana, either as a local dev stack or an overlay alongside an existing chapkit service.

## What you get out of the box

- `GET /metrics` on every scaffolded service - HTTP request counts, latencies, errors, and a few service-level gauges.
- Importing the metrics into Postman / Insomnia: <http://localhost:9090/openapi.json> (the `/metrics` endpoint shows up there too).

## Drop-in compose overlay

Save the following next to your scaffolded `compose.yml` as `compose.monitoring.yml`. It scrapes the model service over the docker network (using its slug as the hostname) and exposes Grafana at <http://localhost:3000>.

```yaml
# compose.monitoring.yml - layer this on top of compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"   # 9091 because chapkit's service uses host port 9090
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prom_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  prom_data:
  grafana_data:
```

Then save the Prometheus config as `monitoring/prometheus.yml` (replace `<your-service-slug>` with the service name from your `compose.yml`):

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: chapkit
    static_configs:
      - targets: ['<your-service-slug>:8000']   # internal docker DNS, container port 8000
```

Bring it all up:

```bash
docker compose -f compose.yml -f compose.monitoring.yml up --build
```

- API: <http://localhost:9090>
- Prometheus: <http://localhost:9091>
- Grafana: <http://localhost:3000> (admin/admin)

## Add the Prometheus data source in Grafana

First time you open Grafana:

1. Configuration -> Data sources -> Add data source -> Prometheus.
2. URL: `http://prometheus:9090` (docker DNS — *not* `localhost`, because Grafana is itself in the docker network).
3. Save & test.

Build dashboards on top of it. Useful starter queries:

- `rate(http_requests_total[1m])` - request rate.
- `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))` - p95 latency.
- `up{job="chapkit"}` - is the service being scraped.

## Production / chap-core deployments

If your model is registered with chap-core, you generally don't need to run a local Prometheus alongside it - chap-core's own observability layer (or the operator's) is what scrapes service metrics in production. The compose overlay above is most useful during local development or for a self-hosted stack that doesn't sit behind chap-core.

## Disabling `/metrics`

If you actively don't want a metrics endpoint exposed (e.g. internal-only deployment with strict surface restrictions), remove the `.with_monitoring()` call from your generated `main.py`:

```python
app = (
    MLServiceBuilder(...)
    # .with_monitoring()   <- remove this line
    .with_registration()
    .build()
)
```

## See also

- [CLI Scaffolding](cli-scaffolding.md) for the chapkit init flow.
- [Deploying to chap-core](deploying-to-chap-core.md) for the production-side compose overlay pattern.
