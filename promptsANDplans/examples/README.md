# Examples Log

> **Purpose:** Working examples — commands, code patterns, API calls, configs — that were proven to work, for reuse.

## Historical records (dated)
- `2026-08-06.md` — Wave 4 examples (E11-E14: CUDA 12.8 fix, Blackwell detection, RDMA avoidance, vision degradation check)

---

## 1. Working SSH pattern (password-based, secret-safe)

```bash
umask 077 && printf 'violin\n' > /tmp/.sshpass_exo
sshpass -f /tmp/.sshpass_exo ssh -o StrictHostKeyChecking=accept-new root@10.2.0.16 'remote command here'
rm -f /tmp/.sshpass_exo
```

## 2. Working detached exo launch (Linux)

```bash
setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda uv run exo > /tmp/exo.log 2>&1 < /dev/null &
```

## 3. Working isolated test node (browser-test backend without touching the cluster)

```bash
rm -rf /tmp/exo-ui-test && mkdir -p /tmp/exo-ui-test && ln -sfn ~/.exo/models /tmp/exo-ui-test/models
cd /Applications/EXO.app/Contents/Resources/exo
EXO_HOME=/tmp/exo-ui-test nohup ./exo --namespace ui-test -m --api-port 52416 --zenoh-port 52424 > /tmp/exo-ui-test.log 2>&1 &
```
> ALWAYS use a non-default API port (52416) — a test node on 52415 squats the app's port and causes exit-code-1 on launch.

## 4. EXO_RUNTIME_DIR hot-swap (point installed app at a fresh bundle, no xcodebuild)

```bash
uv run pyinstaller packaging/pyinstaller/exo.spec
EXO_RUNTIME_DIR=/Users/chris/Documents/GitHub/exo/dist/exo /Applications/EXO.app/Contents/MacOS/EXO
```
> `open` does NOT pass env vars — launch the binary directly or `launchctl setenv EXO_RUNTIME_DIR <path>` first.

## 5. Verify mlx_lm actually got bundled (silent-failure guard)

```bash
grep -c "mlx_lm" build/exo/Analysis-00.toc   # expect > 0 (was 0 when models failed to load)
uv run python -c "import mlx_lm; print(mlx_lm.__file__)"  # must succeed BEFORE pyinstaller
```

## 6. Playwright click-test pattern (proves UI works, captures network)

```js
const captured = [];
page.on("response", r => { if (r.url().includes("/instance/")) captured.push({ method: r.request().method(), url: r.url(), status: r.status() }); });
await page.locator("button", { hasText: "EJECT" }).filter({ hasNotText: "NO" }).first().click();
await page.waitForTimeout(500);
await page.getByRole("button", { name: "YES", exact: true }).first().click();
await page.waitForTimeout(1200);
console.log(captured);  // expect DELETE /instance/<real-id> → 200, NOT /undefined
```

## 3. Working model placement + inference (Linux, single node)

```bash
# Launch (downloads if needed)
curl -X POST http://localhost:52415/place_instance -H "Content-Type: application/json" \
  -d '{"model_id":"mlx-community/Llama-3.2-3B-Instruct-4bit","sharding":"Pipeline","instance_meta":"MlxRing","min_nodes":1,"force_override":false}'

# Chat
curl http://localhost:52415/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```
**Verified results:** Llama-3.2-1B → 73.5 tok/s; Llama-3.2-3B → 29 tok/s

## 4. Working state inspection

```bash
curl -s http://localhost:52415/state | python3 -c "import sys,json; print(list(json.load(sys.stdin).keys()))"
# nodeMemory, nodeBackends, topology, instances, runners, downloads...
```

## 5. Full app rebuild (macOS)

```bash
uv sync --extra mlx
uv sync --reinstall-package exo_rs
cd dashboard && npm run build && cd ..
uv run pyinstaller packaging/pyinstaller/exo.spec --noconfirm
xcodebuild -project app/EXO/EXO.xcodeproj -scheme EXO -configuration Release build
APP=~/Library/Developer/Xcode/DerivedData/EXO-*/Build/Products/Release/EXO.app
sudo rm -rf /Applications/EXO.app && sudo cp -R $APP /Applications/
sudo chmod -R a+rX /Applications/EXO.app
codesign --force --deep --sign - /Applications/EXO.app
open /Applications/EXO.app
```

## 6. Hot-swap dashboard into running app (fast UI iteration)

```bash
DEST="/Applications/EXO.app/Contents/Resources/exo/_internal/dashboard"
sudo cp -R "$DEST" "$DEST.bak-$(date +%Y%m%d-%H%M%S)"
sudo rsync -a --delete /Users/chris/Documents/GitHub/exo/dashboard/build/ "$DEST/"
# refresh browser — done. (Remember: build products go stale; do full rebuild for permanence)
```

## 7. VS Code Insiders patch re-apply

```bash
/Users/chris/scripts/vscode-patch-sensitive-input.sh
# then fully quit (Cmd+Q) and reopen VS Code Insiders
```

## 8. mlx CUDA smoke test (Linux)

```bash
CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda .venv/bin/python -c "
import mlx.core as mx
print('cuda:', mx.cuda.is_available())
a = mx.random.normal((2048, 2048)); b = mx.random.normal((2048, 2048))
mx.eval(a @ b)
print('matmul OK')"
```

## 9. Eject flow (current behavior, user-directed)

1. Click **EJECT** → splits into **NO** (outline) / **YES** (filled)
2. Click **YES** → instance stops
3. **Weights stay. Nothing deleted.** Manual delete via Downloads page.

## 10. Deterministic election check

```bash
curl -s http://localhost:52415/state | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('nodes:', list(d.get('topology',{}).get('nodes',[])))
print('backends:', d.get('nodeBackends',{}))"
# healthy = stable node list, no churn
```

## 7. Passwordless SSH to the Linux GPU box

```bash
# one-time: install pubkey
cat ~/.ssh/id_ed25519.pub | sshpass -f /tmp/.sshpass_exo ssh root@10.2.0.16 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
# now just use the alias (passwordless)
ssh tolinux 'any command'
# tolinux = root@10.2.0.16, id_ed25519 (in ~/.ssh/config)
```

## 8. Verify/repair Linux GPU support in one command

```bash
ssh tolinux 'cd /root/exo && bash scripts/setup_linux_gpu.sh'   # check-only
ssh tolinux 'cd /root/exo && bash scripts/setup_linux_gpu.sh --fix'  # auto-repair
# exit 0 = all 5 checks pass (driver match, pynvml, CUDA headers, MlxCuda advertised)
```
