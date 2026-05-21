import os
import sqlite3
import psycopg2
from psycopg2 import sql
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages connections and schemas for the edge database.
    Supports PostgreSQL with automatic SQLite fallback for local development.
    """
    def __init__(self, db_type="sqlite", pg_conn_str=None):
        self.db_type = db_type.lower()
        self.pg_conn_str = pg_conn_str
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.sqlite_path = self.project_root / "data" / "local_edge.db"
        
        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Creates required directories for local storage and GIS data."""
        dirs = [
            self.project_root / "data" / "raw",
            self.project_root / "data" / "processed",
            self.project_root / "data" / "detections",
            self.project_root / "data" / "legal_zones",
            self.project_root / "models" / "weights"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory verified: {d.relative_to(self.project_root)}")

    def get_connection(self):
        """Returns a database connection based on active configuration."""
        if self.db_type == "postgresql" and self.pg_conn_str:
            try:
                conn = psycopg2.connect(self.pg_conn_str)
                return conn
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")
                self.db_type = "sqlite"
        
        # SQLite connection
        conn = sqlite3.connect(str(self.sqlite_path))
        # Enable foreign keys in SQLite
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_database(self):
        """Initializes tables, relations, and indexes in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        logger.info(f"Initializing {self.db_type.upper()} database schemas...")

        # PostgreSQL Syntax vs SQLite Syntax
        is_pg = self.db_type == "postgresql"
        
        serial_type = "BIGSERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
        t_timestamptz = "TIMESTAMPTZ" if is_pg else "TEXT"
        t_double = "DOUBLE PRECISION" if is_pg else "REAL"
        t_boolean = "BOOLEAN" if is_pg else "INTEGER"
        # Postgress uses 'BYTEA' for raw binary bite, while sqlite uses 'BLOB 
        # This keeps our database schema cross-compatible between local testing and VPS deployment!
        t_blob = "BYTEA" if is_pg else "BLOB"
        now_default = "NOW()" if is_pg else "NOW()"  # Handled on DB-level, but in SQLite we can use DEFAULT CURRENT_TIMESTAMP
        if not is_pg:
            now_default = "CURRENT_TIMESTAMP"

        # 1. Create Telemetry Logs table
        telemetry_table = f"""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id {serial_type},
            timestamp {t_timestamptz} NOT NULL DEFAULT {now_default},
            latitude {t_double} NOT NULL,
            longitude {t_double} NOT NULL,
            altitude_agl {t_double} NOT NULL,
            gimbal_pitch {t_double} NOT NULL,
            gimbal_yaw {t_double} NOT NULL,
            gimbal_roll {t_double} NOT NULL,
            drone_speed {t_double} NOT NULL,
            battery_percentage INTEGER NOT NULL,
            gps_accuracy_m {t_double} NOT NULL
        );
        """
        
        # 2. Create Incidents (Clusters) table
        incidents_table = f"""
        CREATE TABLE IF NOT EXISTS incidents (
            id {serial_type},
            timestamp {t_timestamptz} NOT NULL DEFAULT {now_default},
            centroid_latitude {t_double} NOT NULL,
            centroid_longitude {t_double} NOT NULL,
            severity VARCHAR(20) NOT NULL,
            illegal_zone {t_boolean} NOT NULL DEFAULT 1,
            distance_to_river_m {t_double},
            evidence_image_path VARCHAR(255),
            -- WHAT: New column to store raw binary image bytes directly inside PostgreSQL/SQLite.
            -- WHY: Extracted Jetson snapshots are saved inside the database itself on the VPS!
            evidence_image_blob {t_blob},
            synced_to_cloud {t_boolean} NOT NULL DEFAULT 0
        );
        """

        # 3. Create Detections table with incident_id mapping for cluster isolation and individual filtering
        detections_table = f"""
        CREATE TABLE IF NOT EXISTS detections (
            id {serial_type},
            telemetry_log_id BIGINT,
            incident_id BIGINT,
            timestamp {t_timestamptz} NOT NULL DEFAULT {now_default},
            class_name VARCHAR(50) NOT NULL,
            confidence {t_double} NOT NULL,
            bbox_x_min INTEGER NOT NULL,
            bbox_y_min INTEGER NOT NULL,
            bbox_x_max INTEGER NOT NULL,
            bbox_y_max INTEGER NOT NULL,
            latitude {t_double},
            longitude {t_double},
            frame_path VARCHAR(255),
            FOREIGN KEY (telemetry_log_id) REFERENCES telemetry_logs(id) ON DELETE SET NULL,
            FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE SET NULL
        );
        """

        cursor.execute(telemetry_table)
        cursor.execute(incidents_table)
        cursor.execute(detections_table)

        # ── DYNAMIC COLUMN SCHEMA MIGRATIONS ──────────────────────────────
        # WHAT: Dynamically append columns if database was pre-created before update.
        # WHY: Ensures existing databases don't break with missing column exceptions.
        if self.db_type == "sqlite":
            cursor.execute("PRAGMA table_info(incidents);")
            columns = [col[1] for col in cursor.fetchall()]
            if "evidence_image_blob" not in columns:
                logger.info("🔧 Migrating local SQLite: adding evidence_image_blob column to incidents table...")
                cursor.execute("ALTER TABLE incidents ADD COLUMN evidence_image_blob BLOB;")
                conn.commit()
        else:
            try:
                cursor.execute("ALTER TABLE incidents ADD COLUMN IF NOT EXISTS evidence_image_blob BYTEA;")
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.debug(f"Postgres migration check: {e}")

        # 4. Create indexes for high-performance class filtering & real-time map spatial rendering
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_logs (timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_detections_time ON detections (timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_detections_class ON detections (class_name);",
            "CREATE INDEX IF NOT EXISTS idx_detections_incident ON detections (incident_id);",
            "CREATE INDEX IF NOT EXISTS idx_incidents_coords ON incidents (centroid_latitude, centroid_longitude);",
            "CREATE INDEX IF NOT EXISTS idx_incidents_sync ON incidents (synced_to_cloud);"
        ]

        for idx in indexes:
            cursor.execute(idx)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Edge database initialization complete. Active: {self.db_type.upper()}")

if __name__ == "__main__":
    # Test execution
    manager = DatabaseManager(db_type="sqlite")
    manager.initialize_database()
