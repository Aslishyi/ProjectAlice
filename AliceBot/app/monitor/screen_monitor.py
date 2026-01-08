import threading
import time
import base64
import asyncio
import mss
import mss.tools
import numpy as np
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor


class ScreenMonitor:
    def __init__(self, state_queue, event_loop=None, interval=1.0, diff_threshold=5.0, stability_duration=2.0):
        self.state_queue = state_queue
        self.loop = event_loop
        self.interval = interval
        self.diff_threshold = diff_threshold

        self.running = False
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_hash = None

    def _compress_image(self, img: Image.Image) -> str:
        try:
            # --- 性价比修改 ---

            # 1. 分辨率：1536px
            # (比 1024 清晰得多，能看清文字；比 1920 省 30% Token)
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            buffered = BytesIO()

            # 2. 画质：85 (甚至 90)
            # 既然画质不影响 Token，就开高一点，极大提升 OCR 准确率
            # 消除文字周围的“蚊子噪点”是防止幻觉的关键
            img.convert("RGB").save(buffered, format="JPEG", quality=85, optimize=True)

            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"[Monitor] Compression Error: {e}")
            return None

    def _safe_push(self, event_data):
        """
        线程安全的推送方法：
        如果队列满了，弹出旧的（丢帧），放入新的。
        """
        try:
            if self.state_queue.full():
                try:
                    self.state_queue.get_nowait()  # 丢弃旧帧
                except asyncio.QueueEmpty:
                    pass
            self.state_queue.put_nowait(event_data)
        except Exception as e:
            # 极端情况忽略，防止崩坏
            pass

    def capture_snapshot(self) -> str:
        """主动抓拍 (供用户聊天时调用)"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                # 直接在当前线程压缩
                return self._compress_image(img)
        except Exception as e:
            print(f"[Monitor] Snapshot Error: {e}")
            return None

    def _monitor_loop(self):
        print("[Monitor] Optical Nerve Connected (MSS High-Speed).")

        with mss.mss() as sct:
            monitor = sct.monitors[1]

            while self.running:
                start_time = time.time()
                try:
                    sct_img = sct.grab(monitor)
                    img_array = np.frombuffer(sct_img.rgb, dtype=np.uint8)

                    # 采样哈希，极速去重
                    current_hash = hash(img_array[::100].tobytes())

                    if self.last_hash == current_hash:
                        time.sleep(self.interval)
                        continue

                    self.last_hash = current_hash

                    # 转PIL
                    img_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                    # 提交线程池压缩
                    future = self.executor.submit(self._compress_image, img_pil)
                    img_str = future.result()

                    if img_str:
                        event_data = {
                            "type": "screen_event",
                            "data": img_str,
                            "timestamp": time.time()
                        }
                        # --- 修复核心：使用 _safe_push ---
                        if self.loop and self.loop.is_running():
                            self.loop.call_soon_threadsafe(self._safe_push, event_data)
                        else:
                            self._safe_push(event_data)

                except Exception as e:
                    print(f"[Monitor Loop Error] {e}")

                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.interval - elapsed)
                time.sleep(sleep_time)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.executor.shutdown(wait=False)
