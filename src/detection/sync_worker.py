"""
Offline-First Sync Worker — Jetson Nano Edge Component
Runs as a background thread. Detects network availability and syncs all
unsynced incidents + telemetry to the cloud backend with exponential backoff.

Design: Write-locally-first. Cloud is optional. No data is ever lost on-device.
"""
import time
import logging
import requests
import base64
from pathlib import Path
from threading import Thread, Event
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SyncWorker:
    """
    Resilient background sync agent that:
    - Detects cloud reachability (health-check ping)
    - Drains unsynced incidents from local SQLite/PostgreSQL to cloud
    - Retries with exponential backoff on failure
    - Never blocks the edge pipeline (runs in daemon thread)
    """

    def __init__(self, db_manager, cloud_url: str = "http://localhost:8000",
                 sync_interval_s: float = 5.0):
        self.db_manager     = db_manager
        self.cloud_url      = cloud_url.rstrip("/")
        self.sync_interval  = sync_interval_s
        self._stop_event    = Event()
        self._thread: Optional[Thread] = None
        self._backoff       = 1.0   # seconds, doubles on failure up to max
        self._max_backoff   = 60.0
        self.online         = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the sync worker as a daemon background thread."""
        self._thread = Thread(target=self._run_loop, daemon=True, name="SyncWorker")
        self._thread.start()
        logger.info("Sync worker started (offline-first mode)")

    def stop(self):
        """Signal the worker to stop gracefully."""
        self._stop_event.set()

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _is_cloud_reachable(self) -> bool:
        try:
            r = requests.get(f"{self.cloud_url}/api/stats", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.online = self._is_cloud_reachable()

                if self.online:
                    synced = self._sync_pending_incidents()
                    if synced > 0:
                        logger.info(f"[SyncWorker] Uploaded {synced} pending incident(s) to cloud.")
                    # Reset backoff on success
                    self._backoff = 1.0
                else:
                    # Cloud offline — data is already safe on local DB, just wait
                    logger.debug(f"[SyncWorker] Cloud unreachable. All data stored locally. "
                                 f"Retry in {self._backoff:.0f}s.")
                    self._stop_event.wait(min(self._backoff, self._max_backoff))
                    self._backoff = min(self._backoff * 2, self._max_backoff)
                    continue

            except Exception as e:
                logger.warning(f"[SyncWorker] Unexpected error: {e}")

            self._stop_event.wait(self.sync_interval)

    def _sync_pending_incidents(self) -> int:
        """
        Reads all incidents where synced_to_cloud = 0, uploads them to cloud,
        then marks them synced in the local DB.
        Returns the number of incidents successfully synced.
        """
        conn   = self.db_manager.get_connection()
        cursor = conn.cursor()
        ph     = "?" if self.db_manager.db_type == "sqlite" else "%s"
        synced_count = 0

        try:
            cursor.execute(
                "SELECT id, timestamp, centroid_latitude, centroid_longitude, "
                "severity, illegal_zone, distance_to_river_m, evidence_image_path "
                "FROM incidents WHERE synced_to_cloud = 0 ORDER BY id ASC LIMIT 20"
            )
            rows = cursor.fetchall()

            for row in rows:
                inc_id, ts, lat, lon, severity, illegal, dist, img_path = row

                # Encode evidence image as base64 if it exists on disk
                img_b64 = None
                if img_path:
                    full_path = PROJECT_ROOT / img_path
                    if full_path.exists():
                        with open(full_path, "rb") as f:
                            img_b64 = base64.b64encode(f.read()).decode("utf-8")

                # Fetch linked detections for this incident
                cursor.execute(
                    f"SELECT class_name, confidence, latitude, longitude FROM detections "
                    f"WHERE incident_id = {ph}", (inc_id,)
                )
                det_rows = cursor.fetchall()
                detections = [
                    {"class_name": d[0], "confidence": d[1], "lat": d[2], "lon": d[3]}
                    for d in det_rows
                ]

                payload = {
                    "type": "detections",
                    "payload": {
                        "incident_id":        inc_id,
                        "severity":           severity,
                        "centroid_latitude":  lat,
                        "centroid_longitude": lon,
                        "timestamp":          ts,
                        "illegal_zone":       bool(illegal),
                        "distance_to_river_m": dist,
                        "evidence_image_b64": img_b64,
                        "detections":         detections
                    }
                }

                try:
                    r = requests.post(
                        f"{self.cloud_url}/api/edge/sync",
                        json=payload, timeout=5.0
                    )
                    if r.status_code == 200:
                        # Mark synced in local DB
                        cursor.execute(
                            f"UPDATE incidents SET synced_to_cloud = 1 WHERE id = {ph}", (inc_id,)
                        )
                        conn.commit()
                        synced_count += 1
                except requests.RequestException as e:
                    logger.debug(f"[SyncWorker] Upload failed for incident {inc_id}: {e}")
                    break  # Stop attempting if cloud dropped mid-batch

        except Exception as e:
            logger.error(f"[SyncWorker] DB error during sync: {e}")
        finally:
            cursor.close()
            conn.close()

        return synced_count
