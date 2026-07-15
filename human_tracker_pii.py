#!/usr/bin/env python3

import subprocess
import sys
import os
import cv2
import numpy as np
import time
import math
import socket
import concurrent.futures
import threading
import argparse
import urllib.request
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
from collections import OrderedDict, deque
from ultralytics import YOLO

def run_sys_cmd(cmd_list, **kwargs):
    try:
        return subprocess.run(["sudo"] + cmd_list, **kwargs)
    except FileNotFoundError:
        return subprocess.run(cmd_list, **kwargs)

XIAO_DEFAULT_HOST = "XXX.XXX.XXX.XXX"
CONF = 0.30
MAX_GONE = 25
MATCH_DIST = 90
TRAIL_LEN = 35
WIDTH, HEIGHT = 640, 480
PERSON_CLS = 0

PALETTE = [
    (0, 255, 128), (255, 128, 0), (0, 200, 255),
    (255, 0, 128), (128, 255, 0), (0, 128, 255),
    (255, 255, 0), (128, 0, 255), (0, 255, 255),
    (255, 0, 255),
]


class Tracker:
    def __init__(self):
        self.next_id = 0
        self.objs = OrderedDict()
        self.boxes = OrderedDict()
        self.gone = OrderedDict()
        self.trails = OrderedDict()

    def _add(self, cx, cy, box):
        self.objs[self.next_id] = (cx, cy)
        self.boxes[self.next_id] = box
        self.gone[self.next_id] = 0
        self.trails[self.next_id] = deque(maxlen=TRAIL_LEN)
        self.trails[self.next_id].append((cx, cy))
        self.next_id += 1

    def _remove(self, oid):
        del self.objs[oid]
        del self.boxes[oid]
        del self.gone[oid]
        del self.trails[oid]

    def update(self, dets):
        if not dets:
            for oid in list(self.gone):
                self.gone[oid] += 1
                if self.gone[oid] > MAX_GONE:
                    self._remove(oid)
            return self.boxes

        centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in dets]

        if not self.objs:
            for i, c in enumerate(centers):
                self._add(c[0], c[1], dets[i])
            return self.boxes

        ids = list(self.objs.keys())
        old_pts = list(self.objs.values())

        dist = np.zeros((len(old_pts), len(centers)))
        for i, op in enumerate(old_pts):
            for j, np_ in enumerate(centers):
                dist[i, j] = math.hypot(op[0] - np_[0], op[1] - np_[1])

        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]
        taken_r, taken_c = set(), set()

        for r, c in zip(rows, cols):
            if r in taken_r or c in taken_c:
                continue
            if dist[r, c] > MATCH_DIST:
                continue
            oid = ids[r]
            self.objs[oid] = centers[c]
            self.boxes[oid] = dets[c]
            self.gone[oid] = 0
            self.trails[oid].append(centers[c])
            taken_r.add(r)
            taken_c.add(c)

        for r in set(range(len(old_pts))) - taken_r:
            oid = ids[r]
            self.gone[oid] += 1
            if self.gone[oid] > MAX_GONE:
                self._remove(oid)

        for c in set(range(len(centers))) - taken_c:
            self._add(centers[c][0], centers[c][1], dets[c])

        return self.boxes


def pick_color(tid):
    return PALETTE[tid % len(PALETTE)]

def draw_box(frame, x1, y1, x2, y2, color, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_trail(frame, points, color):
    pts = list(points)
    for i in range(1, len(pts)):
        fade = i / len(pts)
        thick = max(1, int(2 * fade))
        c = tuple(int(v * fade) for v in color)
        cv2.line(frame, pts[i - 1], pts[i], c, thick, cv2.LINE_AA)


def send_command(ip, drive, steer):
    url = f"http://{ip}/track?drive={drive}&steer={steer}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=0.3) as resp:
            return resp.read()
    except Exception:
        pass

def post_ai_frame_async(ip, jpeg_bytes):
    def run():
        url = f"http://{ip}:81/ai_frame"
        try:
            req = urllib.request.Request(url, data=jpeg_bytes, headers={'Content-Type': 'image/jpeg'})
            with urllib.request.urlopen(req, timeout=0.15) as resp:
                resp.read()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('XXX.XXX.XXX.XXX', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = 'XXX.XXX.XXX'
    finally:
        s.close()
    return ip

def check_ip_for_kinebot(ip_candidate):
    url = f"http://{ip_candidate}/status"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=0.2) as resp:
            data = resp.read().decode('utf-8', errors='ignore')
            if "BALANCING" in data or "STALL" in data or "aiEnabled" in data:
                return ip_candidate
    except Exception:
        pass
    return None

def discover_kinebot_on_subnet():
    local_ip = get_local_ip()
    subnets = []

    if local_ip.startswith("XXX.") or local_ip.startswith("10.") or local_ip == "XXX.XXX.XXX.XX":
        print(f"[DISCOVER] Local IP {local_ip} looks like a container/loopback.")
        subnets.extend(["XXX.XXX.XX.", "XXX.XX.XXX.", "XXX.XXX.X.", "XXX.XXX.XX."])
    else:
        parts = local_ip.split('.')
        if len(parts) == 4:
            subnets.append(f"{parts[0]}.{parts[1]}.{parts[2]}.")

    subnets = list(dict.fromkeys(subnets))

    for subnet_prefix in subnets:
        print(f"[DISCOVER] Scanning {subnet_prefix}0/24...")
        ips = [f"{subnet_prefix}{i}" for i in range(1, 255) if f"{subnet_prefix}{i}" != local_ip]

        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            futures = {executor.submit(check_ip_for_kinebot, ip): ip for ip in ips}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    print(f"[DISCOVER] Found KINE-BOT at {res}")
                    return res
    return None

def resolve_xiao_host(host):
    if host == "kinebot.local":
        print("[NET] Resolving kinebot.local via mDNS...")
        try:
            ip = socket.gethostbyname(host)
            print(f"[NET] Resolved to {ip}")
            return ip
        except socket.gaierror:
            print("[NET] mDNS failed, scanning subnet...")
            discovered = discover_kinebot_on_subnet()
            if discovered:
                return discovered
            print("[NET] Scan failed, falling back to XXX.XXX.XXX.XXX")
            return "XXX.XXX.XXX.XXX"  # Replace with actual IP
    return host

def ensure_wifi_connection():
    target_ssid = "UR_NETWORK"
    target_pass = "UR_PASSWORD"

    print("[WIFI] Checking connection...")
    current_ssid = None

    try:
        current_ssid = subprocess.check_output(["iwgetid", "-r"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    if not current_ssid:
        try:
            out = subprocess.check_output(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"], stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if line.startswith("yes:"):
                    current_ssid = line.split(":", 1)[1]
                    break
        except Exception:
            pass

    if current_ssid == target_ssid:
        print(f"[WIFI] Already on {target_ssid}")
        return

    if current_ssid:
        print(f"[WIFI] On different network: '{current_ssid}'")
    else:
        print("[WIFI] Not connected.")

    print(f"[WIFI] Connecting to '{target_ssid}'...")

    try:
        res = run_sys_cmd(["nmcli", "dev", "wifi", "connect", target_ssid, "password", target_pass],
                           capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print(f"[WIFI] Connected to '{target_ssid}'")
            time.sleep(2)
            return
        else:
            print(f"[WIFI] nmcli failed: {res.stderr.strip() if res.stderr else 'unknown'}")
    except Exception as e:
        print(f"[WIFI] nmcli error: {e}")

    try:
        net_id = subprocess.check_output(["sudo", "wpa_cli", "add_network"], stderr=subprocess.DEVNULL).decode().strip()
        if net_id.isdigit():
            subprocess.run(["sudo", "wpa_cli", "set_network", net_id, "ssid", f'"{target_ssid}"'], stdout=subprocess.DEVNULL)
            subprocess.run(["sudo", "wpa_cli", "set_network", net_id, "psk", f'"{target_pass}"'], stdout=subprocess.DEVNULL)
            subprocess.run(["sudo", "wpa_cli", "enable_network", net_id], stdout=subprocess.DEVNULL)
            subprocess.run(["sudo", "wpa_cli", "select_network", net_id], stdout=subprocess.DEVNULL)
            subprocess.run(["sudo", "wpa_cli", "save_config"], stdout=subprocess.DEVNULL)
            print(f"[WIFI] wpa_cli configured '{target_ssid}'")
            time.sleep(5)
            return
    except Exception as e:
        print(f"[WIFI] wpa_cli failed: {e}")

    print(f"[WIFI] Couldn't connect automatically. Please join '{target_ssid}' manually.")


latest_jpeg_bytes = None
frame_lock = threading.Lock()
XIAO_IP = ""

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KINE-BOT — Uno Q AI Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
body {
  background: #090d16;
  color: #f0f6fc;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
  padding: 20px;
  overflow-x: hidden;
}
h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin-bottom: 20px;
  background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 1px;
  text-shadow: 0 0 20px rgba(79, 172, 254, 0.3);
}
.c {
  width: 100%;
  max-width: 480px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.card {
  background: rgba(21, 26, 38, 0.6);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.ct {
  font-size: 1.2rem;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.badge {
  font-size: 0.8rem;
  padding: 4px 10px;
  border-radius: 12px;
  font-weight: 600;
  text-transform: uppercase;
}
.b-stall { background: rgba(255, 42, 133, 0.15); color: #ff2a85; border: 1px solid rgba(255, 42, 133, 0.3); }
.b-bal { background: rgba(0, 242, 254, 0.15); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.3); }
.vid-wrap {
  position: relative;
  border-radius: 14px;
  overflow: hidden;
  background: #0a0e18;
  line-height: 0;
  border: 1px solid rgba(255, 255, 255, 0.04);
}
.vid-wrap img {
  width: 100%;
  display: block;
  min-height: 200px;
  background: #0a0e18;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 15px;
}
.m {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.04);
  border-radius: 14px;
  padding: 12px;
  text-align: center;
}
.mv { font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-top: 4px; }
.ml { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.pad {
  display: grid;
  grid-template-areas:
    ". u ."
    "l c r"
    ". d .";
  grid-template-columns: repeat(3, 75px);
  grid-template-rows: repeat(3, 75px);
  gap: 12px;
  justify-content: center;
  margin: 15px 0;
}
.btn {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  color: #f0f6fc;
  font-size: 1.4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s;
}
.btn:hover { background: rgba(255, 255, 255, 0.08); }
.btn:active, .btn.on {
  background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
  border-color: #00f2fe;
  color: #090d16;
  transform: scale(0.92);
}
.bu { grid-area: u; }
.bl { grid-area: l; }
.br { grid-area: r; }
.bd { grid-area: d; }
.bc { grid-area: c; border-radius: 50%; color: #ff2a85; background: rgba(255, 42, 133, 0.05); border-color: rgba(255, 42, 133, 0.2); }
.bc:active, .bc.on { background: #ff2a85; color: #fff; }
</style>
</head>
<body>
<h1>KINE-BOT AI</h1>
<div class="c">
  <div class="card">
    <div class="ct">Annotated Stream <span class="badge b-bal">AI ACTIVE</span></div>
    <div class="vid-wrap">
      <img id="stream" src="/stream" alt="AI Feed">
    </div>
  </div>
  <div class="card">
    <div class="ct">Telemetry <span id="badge" class="badge b-stall">STALL</span></div>
    <div class="grid">
      <div class="m"><div class="ml">Tilt</div><div class="mv" id="ang">0.0&deg;</div></div>
      <div class="m"><div class="ml">Command</div><div class="mv" id="cmd">NONE</div></div>
      <div class="m"><div class="ml">Source</div><div class="mv" id="src">AI</div></div>
      <div class="m"><div class="ml">State</div><div class="mv" id="state">STALL</div></div>
    </div>
  </div>
  <div class="card">
    <div class="ct">Manual Override Controls</div>
    <div class="pad">
      <div class="btn bu" id="bF">&#x25B2;</div>
      <div class="btn bl" id="bL">&#x25C0;</div>
      <div class="btn bc" id="bS">STOP</div>
      <div class="btn br" id="bR">&#x25B6;</div>
      <div class="btn bd" id="bB">&#x25BC;</div>
    </div>
  </div>
</div>
<script>
var keys = { w: false, a: false, s: false, d: false, arrowup: false, arrowdown: false, arrowleft: false, arrowright: false };

function getDriveCmd() {
  if (keys.w || keys.arrowup) return "F";
  if (keys.s || keys.arrowdown) return "B";
  return "S";
}
function getSteerCmd() {
  if (keys.a || keys.arrowleft) return "L";
  if (keys.d || keys.arrowright) return "R";
  return "S";
}
function setKeyState(key, pressed) {
  var k = key.toLowerCase();
  if (keys[k] !== undefined) {
    if (keys[k] === pressed) return;
    keys[k] = pressed;
    updateButtonStyles();
    sendControl();
  }
}
function updateButtonStyles() {
  var drive = getDriveCmd();
  var steer = getSteerCmd();
  document.getElementById("bF").classList.toggle("on", drive === "F");
  document.getElementById("bB").classList.toggle("on", drive === "B");
  document.getElementById("bL").classList.toggle("on", steer === "L");
  document.getElementById("bR").classList.toggle("on", steer === "R");
  document.getElementById("bS").classList.toggle("on", drive === "S" && steer === "S");
}
function sendControl() {
  var x = new XMLHttpRequest();
  x.open("GET", "/move?drive=" + getDriveCmd() + "&steer=" + getSteerCmd(), true);
  x.send();
}
window.addEventListener("keydown", function(e) { setKeyState(e.key, true); });
window.addEventListener("keyup", function(e) { setKeyState(e.key, false); });

function bind(id, key) {
  var el = document.getElementById(id);
  if (el) {
    el.addEventListener("mousedown", function(e) { e.preventDefault(); setKeyState(key, true); });
    el.addEventListener("touchstart", function(e) { e.preventDefault(); setKeyState(key, true); });
    el.addEventListener("mouseup", function(e) { e.preventDefault(); setKeyState(key, false); });
    el.addEventListener("touchend", function(e) { e.preventDefault(); setKeyState(key, false); });
  }
}
bind("bF", "w"); bind("bB", "s"); bind("bL", "a"); bind("bR", "d");
document.getElementById("bS").addEventListener("mousedown", function() {
  for (var k in keys) keys[k] = false;
  updateButtonStyles();
  sendControl();
});

setInterval(function() {
  var x = new XMLHttpRequest();
  x.open("GET", "/telemetry", true);
  x.onload = function() {
    if (x.status === 200) {
      try {
        var j = JSON.parse(x.responseText);
        document.getElementById("ang").innerText = j.angle.toFixed(1) + "\u00b0";
        document.getElementById("cmd").innerText = j.cmd;
        document.getElementById("src").innerText = j.source;
        document.getElementById("state").innerText = j.state;
        var b = document.getElementById("badge");
        b.innerText = j.state;
        b.className = "badge " + (j.state === "BALANCING" ? "b-bal" : "b-stall");
      } catch(e) {}
    }
  };
  x.send();
}, 250);
</script>
</body>
</html>
"""

class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global latest_jpeg_bytes, XIAO_IP
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                last_sent_bytes = None
                while True:
                    with frame_lock:
                        frame_bytes = latest_jpeg_bytes

                    if frame_bytes is None:
                        time.sleep(1.0)
                        continue

                    if frame_bytes is last_sent_bytes:
                        time.sleep(0.03)
                        continue

                    self.wfile.write(b'\r\n--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('utf-8'))
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                    last_sent_bytes = frame_bytes
                    time.sleep(0.05)
            except Exception:
                pass
        elif self.path.startswith('/move'):
            query = parse_qs(urlparse(self.path).query)
            drive = query.get('drive', ['S'])[0]
            steer = query.get('steer', ['S'])[0]
            send_command(XIAO_IP, drive, steer)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        elif self.path == '/telemetry':
            url = f"http://{XIAO_IP}/status"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=0.25) as resp:
                    data = resp.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception:
                pass
            self.send_response(500)
            self.end_headers()

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def start_web_server(port, xiao_ip):
    global XIAO_IP
    XIAO_IP = xiao_ip
    server = ThreadingHTTPServer(('0.0.0.0', port), StreamingHandler)
    local_ip = get_local_ip()
    print(f"[SERVER] Dashboard running at http://{local_ip}:{port}")
    try:
        server.serve_forever()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="KINE-BOT AI Tracker")
    parser.add_argument("--ip", type=str, default=XIAO_DEFAULT_HOST)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--no-display", action="store_true", default=True)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if args.display:
        args.no_display = False

    model_path = args.model
    if not model_path:
        for candidate in ["yolov8n.pt", "yolov8s.pt", "yolov8x.pt"]:
            if os.path.exists(candidate):
                model_path = candidate
                break
        if not model_path:
            model_path = "yolov8n.pt"

    print(f"[AI] Loading {model_path}...")
    model = YOLO(model_path)
    print("[AI] Model ready.")

    ensure_wifi_connection()
    ip = resolve_xiao_host(args.ip)
    print(f"[NET] Target: {ip}")

    threading.Thread(target=start_web_server, args=(8080, ip), daemon=True).start()

    stream_url = f"http://{ip}:81/stream"
    print(f"[CAM] Connecting to {stream_url}...")
    stream = None

    for attempt in range(10):
        try:
            stream = urllib.request.urlopen(stream_url, timeout=5.0)
            print("[CAM] Connected!")
            break
        except Exception as e:
            print(f"[CAM] Attempt {attempt + 1}/10 failed: {e}")
            time.sleep(3)

    if not stream:
        print("[FATAL] No camera stream. Exiting.")
        sys.exit(1)

    tracker = Tracker()
    buf = bytes()
    last_cmd_time = 0.0
    last_drive = "S"
    last_steer = "S"
    is_driving = False

    global latest_jpeg_bytes

    CENTER_X = WIDTH // 2
    STEER_DEADBAND = 60
    DIST_CLOSE = 220
    DIST_FAR = 180

    fps_time = time.time()
    frame_count = 0
    fps = 0.0
    headless = args.no_display

    print("[RUN] Tracking started. Ctrl+C to stop.")

    try:
        while True:
            try:
                chunk = stream.read(4096)
                if not chunk:
                    break
                buf += chunk
            except Exception:
                break

            a = buf.find(b'\xff\xd8')
            b = buf.find(b'\xff\xd9')
            if a == -1 or b == -1 or b <= a:
                continue

            jpg = buf[a:b + 2]
            buf = buf[b + 2:]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            results = model(frame, verbose=False, conf=CONF, imgsz=320)
            detections = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == PERSON_CLS:
                        detections.append(box.xyxy[0].cpu().numpy().astype(int))

            tracked = tracker.update(detections)

            drive_cmd = "S"
            steer_cmd = "S"
            target_id = -1
            max_area = 0

            for tid, bbox in tracked.items():
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_id = tid

            if target_id != -1:
                x1, y1, x2, y2 = tracked[target_id]
                cx = (x1 + x2) // 2
                h_box = y2 - y1

                dx = cx - CENTER_X
                if dx < -STEER_DEADBAND:
                    steer_cmd = "L"
                elif dx > STEER_DEADBAND:
                    steer_cmd = "R"

                if is_driving:
                    if h_box >= DIST_CLOSE:
                        is_driving = False
                    else:
                        drive_cmd = "F"
                else:
                    if h_box < DIST_FAR:
                        is_driving = True
                        drive_cmd = "F"

                color = pick_color(target_id)
                draw_box(frame, x1, y1, x2, y2, color, f"ID {target_id}")
                if target_id in tracker.trails:
                    draw_trail(frame, tracker.trails[target_id], color)
                cv2.circle(frame, (cx, (y1 + y2) // 2), 5, (0, 0, 255), -1)
            else:
                is_driving = False

            cv2.rectangle(frame, (8, 8), (260, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Target: {target_id if target_id != -1 else 'NONE'}", (16, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"D:{drive_cmd}  S:{steer_cmd}", (16, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            try:
                _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with frame_lock:
                    latest_jpeg_bytes = jpeg_buf.tobytes()
            except Exception:
                pass

            now = time.time()
            if (drive_cmd != last_drive or steer_cmd != last_steer) or (now - last_cmd_time > 0.2):
                send_command(ip, drive_cmd, steer_cmd)
                last_drive = drive_cmd
                last_steer = steer_cmd
                last_cmd_time = now

            frame_count += 1
            if now - fps_time >= 1.0:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            if not headless:
                cv2.imshow("KINE-BOT Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count == 0:
                    print(f"[TRACK] FPS={fps:.1f} target={target_id} drive={drive_cmd} steer={steer_cmd}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        send_command(ip, "S", "S")
        if stream:
            stream.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()