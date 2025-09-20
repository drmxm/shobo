# On Jetson host:
xhost +local:root

docker compose build
docker compose up

docker compose up --no-build perception




docker compose stop perception || true
docker compose rm -f -s -v perception
docker compose build --no-cache perception
xhost +local:root   # on the host, once per session
docker compose up --no-build perception

