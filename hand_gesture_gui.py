#!/usr/bin/env python3
# hand_gesture_gui.py
# Final single-file Hand Gesture project:
# - GUI (Tkinter)
# - MediaPipe hand tracking
# - Mouse control (index finger), pinch left-click, right-click via L-sign
# - Scrolling (open palm + vertical movement)
# - Volume control via pycaw (robust fallback if unavailable)
# - Keyboard shortcuts mapped to gestures
# - Debounce / cooldowns for actions
# Works on Windows. Tweak thresholds at top as needed.

import time
import threading
import math
import sys
import traceback

import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Button

# MediaPipe
import mediapipe as mp

# Try import pycaw; handle gracefully if not present or fails
try:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL, CoInitialize
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYAUDIO_AVAILABLE = True
except Exception:
    PYAUDIO_AVAILABLE = False

# -------------------------
# Configuration (tweakable)
# -------------------------
PINCH_THRESHOLD = 0.04        # normalized distance (thumb-index) for left click
DIST_OK_THRESHOLD = 0.035    # normalized thumb-index for OK sign
SMOOTHING = 0.25             # mouse smoothing factor (0..1)
SCROLL_SENSITIVITY = 900     # multiplier to convert normalized dy to scroll units
SCROLL_MIN_DELTA = 0.004     # minimal normalized y-delta to start scrolling
COOLDOWN_SHORT = 0.35        # secs (volume)
COOLDOWN_MED = 0.9           # secs (clicks/keyboard/mute)
COOLDOWN_SCROLL = 0.06       # secs between scroll events
CAM_W, CAM_H = 640, 480      # camera capture resolution
DISPLAY_W, DISPLAY_H = 880, 520  # GUI display size for showing frames

pyautogui.FAILSAFE = False   # disable fail-safe to avoid exceptions when mouse goes to corner

# -------------------------
# Volume Controller (robust)
# -------------------------
class VolumeController:
    def __init__(self):
        self.volume = None
        if not PYAUDIO_AVAILABLE:
            print("[VOL] pycaw/comtypes not available — volume control disabled.")
            return
        try:
            # Initialize COM for this thread
            try:
                CoInitialize()
            except Exception:
                # some environments may already be initialized; ignore
                pass
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            print("[VOL] pycaw initialized — volume control enabled.")
        except Exception as e:
            print("[VOL] pycaw initialization failed — volume control disabled.")
            print("       Error:", e)
            self.volume = None

    def set_volume(self, level: float):
        if self.volume:
            level = max(0.0, min(1.0, level))
            try:
                self.volume.SetMasterVolumeLevelScalar(level, None)
            except Exception:
                pass

    def get_volume(self) -> float:
        if self.volume:
            try:
                return float(self.volume.GetMasterVolumeLevelScalar())
            except Exception:
                return 0.0
        return 0.0

    def volume_up(self, step=0.05):
        if self.volume:
            v = self.get_volume()
            self.set_volume(min(v + step, 1.0))

    def volume_down(self, step=0.05):
        if self.volume:
            v = self.get_volume()
            self.set_volume(max(v - step, 0.0))

    def mute_unmute(self):
        if self.volume:
            try:
                self.volume.SetMute(not self.volume.GetMute(), None)
            except Exception:
                pass

# -------------------------
# Gesture Detector (MediaPipe)
# -------------------------
class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # reasonable defaults
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def finger_up(tip, pip):
        return tip.y < pip.y

    @staticmethod
    def euclidean(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def classify(self, lm):
        # lm: list of 21 landmarks
        thumb_tip, thumb_ip = lm[4], lm[3]
        index_tip, index_pip = lm[8], lm[6]
        middle_tip, middle_pip = lm[12], lm[10]
        ring_tip, ring_pip = lm[16], lm[14]
        pinky_tip, pinky_pip = lm[20], lm[18]

        thumb_up = thumb_tip.x < thumb_ip.x  # typical mirrored webcam assumption
        index_up = self.finger_up(index_tip, index_pip)
        middle_up = self.finger_up(middle_tip, middle_pip)
        ring_up = self.finger_up(ring_tip, ring_pip)
        pinky_up = self.finger_up(pinky_tip, pinky_pip)

        fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])
        dist_thumb_index = self.euclidean(thumb_tip, index_tip)

        # Priority-based rules
        # Open Palm
        if fingers_up_count == 4 and thumb_up:
            return "OPEN_PALM"

        # OK sign (thumb and index touching)
        if dist_thumb_index < DIST_OK_THRESHOLD:
            return "OK_SIGN"

        # Peace (V)
        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE"

        # Rock (index + pinky)
        if index_up and not middle_up and not ring_up and pinky_up:
            return "ROCK"

        # L sign (thumb + index up)
        if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            return "L_SIGN"

        # Thumbs up
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return "THUMBS_UP"

        # Fist
        if fingers_up_count == 0 and not thumb_up:
            return "FIST"

        return "UNKNOWN"

# -------------------------
# Main GUI App
# -------------------------
class GestureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Gesture — Volume + Mouse + Scroll + Shortcuts")
        self.window.geometry("940x780")

        # Detector & volume
        self.detector = GestureDetector()
        self.volume = VolumeController()

        # GUI elements
        self.video_label = Label(window)
        self.video_label.pack(padx=6, pady=8)

        self.status_label = Label(window, text="Status: Idle", font=("Arial", 14))
        self.status_label.pack(pady=6)

        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=6)
        self.start_btn = Button(btn_frame, text="Start", width=16, command=self.start, bg="#4CAF50", fg="white")
        self.start_btn.grid(row=0, column=0, padx=8)
        self.stop_btn = Button(btn_frame, text="Stop", width=16, command=self.stop, bg="#F44336", fg="white")
        self.stop_btn.grid(row=0, column=1, padx=8)

        # runtime state
        self.running = False
        self.cap = None
        self.prev_mouse_x = None
        self.prev_mouse_y = None
        self.screen_w, self.screen_h = pyautogui.size()
        self.last_times = {
            "volume": 0.0,
            "mute": 0.0,
            "click": 0.0,
            "keyboard": 0.0,
            "scroll": 0.0
        }
        self.scroll_mode = False
        self.last_index_y = None

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
            self.status_label.config(text="Status: Running")

    def stop(self):
        self.running = False
        self.status_label.config(text="Status: Stopped")
        # camera release handled in loop

    def _throttle(self, key, cooldown):
        now = time.time()
        if now - self.last_times.get(key, 0) >= cooldown:
            self.last_times[key] = now
            return True
        return False

    def _perform_keyboard_shortcut(self, gesture):
        if not self._throttle("keyboard", COOLDOWN_MED):
            return
        try:
            if gesture == "OK_SIGN":
                pyautogui.hotkey('ctrl', 'c')  # Copy
                print("[KB] Ctrl+C (Copy)")
            elif gesture == "L_SIGN":
                pyautogui.hotkey('ctrl', 'v')  # Paste
                print("[KB] Ctrl+V (Paste)")
            elif gesture == "PEACE":
                pyautogui.hotkey('ctrl', 's')  # Save
                print("[KB] Ctrl+S (Save)")
        except Exception as e:
            print("[KB] Shortcut failed:", e)

    def _loop(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        except Exception:
            self.cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = None
            try:
                results = self.detector.hands.process(rgb)
            except Exception:
                # sometimes MediaPipe throws internal errors — ignore and continue
                results = None

            gesture_text = "No hand"

            index_pos = None
            thumb_pos = None

            if results and results.multi_hand_landmarks:
                handlm = results.multi_hand_landmarks[0]
                self.detector.mp_draw.draw_landmarks(frame, handlm, self.detector.mp_hands.HAND_CONNECTIONS)
                lm = handlm.landmark

                gesture_text = self.detector.classify(lm)

                index_tip = lm[8]
                thumb_tip = lm[4]
                index_pos = (index_tip.x, index_tip.y)
                thumb_pos = (thumb_tip.x, thumb_tip.y)

                # ----- Mouse Move (index finger) -----
                target_x = int(index_tip.x * self.screen_w)
                target_y = int(index_tip.y * self.screen_h)
                if self.prev_mouse_x is None:
                    self.prev_mouse_x, self.prev_mouse_y = target_x, target_y

                smooth_x = int(self.prev_mouse_x + (target_x - self.prev_mouse_x) * SMOOTHING)
                smooth_y = int(self.prev_mouse_y + (target_y - self.prev_mouse_y) * SMOOTHING)
                try:
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                except Exception:
                    pass
                self.prev_mouse_x, self.prev_mouse_y = smooth_x, smooth_y

                # ----- Left Click via Pinch -----
                if thumb_pos and index_pos:
                    dist_pin = math.hypot(thumb_pos[0] - index_pos[0], thumb_pos[1] - index_pos[1])
                    if dist_pin < PINCH_THRESHOLD and self._throttle("click", COOLDOWN_MED):
                        try:
                            pyautogui.click()
                            print("[MOUSE] Left Click (Pinch)")
                        except Exception:
                            pass

                # ----- Right Click via L_SIGN -----
                if gesture_text == "L_SIGN" and self._throttle("click", COOLDOWN_MED):
                    try:
                        pyautogui.rightClick()
                        print("[MOUSE] Right Click (L sign)")
                    except Exception:
                        pass

                # ----- Volume Controls -----
                if gesture_text == "THUMBS_UP" and self._throttle("volume", COOLDOWN_SHORT):
                    self.volume.volume_up()
                    print("[VOL] Up ->", round(self.volume.get_volume(), 2) if self.volume.volume else "n/a")

                if gesture_text == "FIST" and self._throttle("volume", COOLDOWN_SHORT):
                    self.volume.volume_down()
                    print("[VOL] Down ->", round(self.volume.get_volume(), 2) if self.volume.volume else "n/a")

                if gesture_text == "ROCK" and self._throttle("mute", COOLDOWN_MED):
                    self.volume.mute_unmute()
                    print("[VOL] Toggle Mute")

                # ----- Keyboard Shortcuts -----
                if gesture_text in ("OK_SIGN", "L_SIGN", "PEACE"):
                    self._perform_keyboard_shortcut(gesture_text)

                # ----- Scrolling mode (OPEN_PALM) -----
                if gesture_text == "OPEN_PALM":
                    if not self.scroll_mode:
                        self.scroll_mode = True
                        self.last_index_y = index_tip.y
                        print("[SCROLL] Enter scroll mode")
                    else:
                        dy = self.last_index_y - index_tip.y  # positive -> moved up
                        if abs(dy) > SCROLL_MIN_DELTA and self._throttle("scroll", COOLDOWN_SCROLL):
                            scroll_amount = int(dy * SCROLL_SENSITIVITY)
                            if scroll_amount != 0:
                                try:
                                    pyautogui.scroll(scroll_amount)
                                    # print("[SCROLL] scroll", scroll_amount)
                                except Exception:
                                    pass
                        # update baseline gradually
                        self.last_index_y = self.last_index_y * 0.85 + index_tip.y * 0.15
                else:
                    if self.scroll_mode:
                        self.scroll_mode = False
                        self.last_index_y = None
                        print("[SCROLL] Exit scroll mode")

            # Prepare and show display in GUI
            try:
                display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            except Exception:
                display = frame
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
            # attach to label
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
            self.status_label.config(text=f"Gesture: {gesture_text}")

        # cleanup
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.prev_mouse_x = None
        self.prev_mouse_y = None

# -------------------------
# Run App
# -------------------------
def main():
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
        sys.exit(1)
