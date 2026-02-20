# Fix "permission denied" when removing Docker container

Your system has **Docker installed twice** (Snap + apt). That often causes this error.

## Option A: Remove Snap Docker and fix (recommended)

Run these in your terminal, one by one:

```bash
# 1. Remove Docker from Snap (you still have docker from apt)
sudo snap remove docker --purge

# 2. Remove conflicting AppArmor profiles
sudo aa-remove-unknown

# 3. Restart Docker (socket + service)
sudo systemctl restart docker.socket docker.service

# 4. Remove the stuck container
sudo docker rm -f a28b8924e399
```

After this, use Docker only from apt (`docker.io`). If `docker` command is missing, run: `sudo apt install docker.io`.

---

## Option B: Nuclear option (if A doesn't work)

Stop Docker, delete the container from disk, start Docker:

```bash
sudo systemctl stop docker
sudo rm -rf /var/lib/docker/containers/a28b8924e399e23151a9434ff7639a3f6934081aae04dcb13dee6f9f539544c1
sudo systemctl start docker
```

Then bring your app back up with:

```bash
cd /home/derfmeow/Desktop/pdfreader
docker compose up -d
```
