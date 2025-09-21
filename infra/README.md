# Infra (Foxglove / Telemetry)

Runs:
- **foxglove_bridge** — WebSocket server for Foxglove Studio (default `ws://0.0.0.0:8765`)
- **rosbridge_websocket** (optional) — JSON-over-WS at `ws://0.0.0.0:9090` for quick scripts

## Env Vars
- `FOXGLOVE_PORT` (default `8765`)
- `FOXGLOVE_ADDRESS` (default `0.0.0.0`)
- `ENABLE_ROSBRIDGE` (`true`/`false`, default `false`)
- `ROSBRIDGE_PORT` (default `9090`)

## Connect from Foxglove Studio
1. Open Foxglove Studio.
2. Click **+ Add connection** → **Foxglove WebSocket**.
3. Enter `ws://<JETSON_IP>:8765` (or another port if you changed it).
4. You should see topics like `/sensors/uvc/image_raw`.

If images don’t appear:
- Verify the camera publisher is running and `/sensors/uvc/image_raw` exists.
- Check QoS: set BestEffort/Volatile in `params/foxglove.params.yaml` for image topics.
- Inspect container logs: `docker logs -f xshan-infra`.


### from xshan/deploy/compose rebuild and redeploy
docker compose build detector --no-cache --pull
docker compose up -d --force-recreate detector