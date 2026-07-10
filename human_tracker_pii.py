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

XIAO_DEFAULT_HOST = "192.168.31.138"
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
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
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

    if local_ip.startswith("172.") or local_ip.startswith("10.") or local_ip == "127.0.0.1":
        print(f"[DISCOVER] Local IP {local_ip} looks like a container/loopback.")
        subnets.extend(["192.168.31.", "192.168.1.", "192.168.0.", "192.168.4."])
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
            print("[NET] Scan failed, falling back to 192.168.4.1")
            return "192.168.4.1"
    return host

def ensure_wifi_connection():
    target_ssid = "FAST_AND_FURIOUS"
    target_pass = "ravi@7341"

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
        print(f"[WIFI] On differe