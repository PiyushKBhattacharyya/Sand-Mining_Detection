import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
import sys

# Add preprocess directory to import db_setup
sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocess"))
from db_setup import DatabaseManager
from zone_builder import build_buffer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusterEngine:
    """
    Groups spatial detections into clusters using DBSCAN,
    calculates cluster severities, and verifies violations against the 500m legal buffer.
    """
    def __init__(self, db_manager: DatabaseManager, buffer_path: str = None):
        self.db_manager = db_manager
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.buffer_path = buffer_path or (
            self.project_root / "data" / "legal_zones" / "river_buffer_1km.geojson"
        )
        
        # Load the pre-calculated 1km illegal zone polygon
        self.illegal_zone_gdf = None
        self._load_buffer_zone()

    def _load_buffer_zone(self):
        """Loads the river buffer GeoJSON for fast spatial contains queries."""
        if not Path(self.buffer_path).exists():
            logger.warning(f"River buffer geojson not found at: {self.buffer_path}. Real-time zone check will assume all points inside.")
            return
        
        try:
            self.illegal_zone_gdf = gpd.read_file(self.buffer_path)
            # Ensure it is in EPSG:4326
            if self.illegal_zone_gdf.crs != "EPSG:4326":
                self.illegal_zone_gdf = self.illegal_zone_gdf.to_crs("EPSG:4326")
            logger.info("Successfully loaded 1km River Buffer Zone.")
        except Exception as e:
            logger.error(f"Error loading buffer zone geometry: {e}")

    def set_radius(self, radius_m: float):
        """
        Hot-reload the enforcement zone with a new radius (metres).
        Called when the operator changes the slider on the dashboard.
        Rebuilds the GeoJSON file then reloads the Shapely geometry in-memory
        so all subsequent is_in_illegal_zone() calls use the new boundary.
        """
        logger.info(f"ClusterEngine: hot-reloading buffer at {radius_m:.0f}m")
        ok = build_buffer(radius_m=radius_m)
        if ok:
            self._load_buffer_zone()   # reload Shapely geometry from updated file
            logger.info(f"Zone enforcement updated to {radius_m:.0f}m")
        else:
            logger.warning("Buffer rebuild failed — keeping previous zone geometry")

    def is_in_illegal_zone(self, lat: float, lon: float) -> bool:
        """Returns True if the coordinate is inside the 500m buffer zone."""
        if self.illegal_zone_gdf is None:
            return True # Fallback to warning if no file exists
            
        point = Point(lon, lat)  # Shapely uses (x, y) = (lon, lat)
        # Check if the point is within any geometry in the buffer GeoDataFrame
        contains = self.illegal_zone_gdf.geometry.contains(point).any()
        return bool(contains)

    def calculate_severity(self, detections: List[Dict[str, Any]]) -> str:
        """
        Calculates cluster severity based on constituent detections:
        - CRITICAL: JCB + Truck + Workers co-located inside illegal zone.
        - HIGH: Heavy machinery (JCB or Truck) + Workers present.
        - MEDIUM: Vehicles only, no human workers detected.
        - LOW: Workers/people only, no excavation machinery.
        """
        classes = [d['class_name'].lower() for d in detections]
        
        has_jcb = 'jcb' in classes
        has_truck = 'truck' in classes
        has_person = 'person' in classes
        
        if has_jcb and has_truck and has_person:
            return "CRITICAL"
        elif (has_jcb or has_truck) and has_person:
            return "HIGH"
        elif has_jcb or has_truck:
            return "MEDIUM"
        else:
            return "LOW"

    def cluster_detections(self, detections: List[Dict[str, Any]], eps_meters: float = 50.0, min_samples: int = 1) -> List[Dict[str, Any]]:
        """
        Clusters active detections using DBSCAN.
        Assigns cluster IDs and calculates incident details.
        
        Args:
            detections: List of detection dicts containing 'lat', 'lon', 'class_name', 'confidence', etc.
            eps_meters: Maximum distance between points to be grouped in same cluster.
            min_samples: Minimum points in a group to form a cluster.
            
        Returns:
            List of generated Incident dicts ready for database logging.
        """
        if not detections:
            return []
            
        # Extract coordinates
        coords = np.array([[d['lat'], d['lon']] for d in detections])
        
        # Approximate meters to degrees for DBSCAN (1 deg ~ 111,320m)
        eps_deg = eps_meters / 111320.0
        
        # Fit DBSCAN
        db = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
        labels = db.labels_
        
        # Group detections by cluster labels
        clusters = {}
        for idx, label in enumerate(labels):
            # In DBSCAN, -1 is noise, but since we set min_samples=1, we don't have noise.
            # However, if min_samples > 1, we log noise as individual isolated incidents
            clusters.setdefault(label, []).append(detections[idx])
            
        incidents = []
        for label, cluster_detections in clusters.items():
            # Calculate centroid
            cluster_coords = np.array([[d['lat'], d['lon']] for d in cluster_detections])
            centroid_lat = float(np.mean(cluster_coords[:, 0]))
            centroid_lon = float(np.mean(cluster_coords[:, 1]))
            
            # Check illegal zone
            illegal = self.is_in_illegal_zone(centroid_lat, centroid_lon)
            
            # Determine severity
            severity = self.calculate_severity(cluster_detections)
            
            incident = {
                'label': label,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'severity': severity,
                'illegal_zone': 1 if illegal else 0,
                'detections': cluster_detections
            }
            incidents.append(incident)
            
        return incidents

    def save_incidents_to_db(self, incidents: List[Dict[str, Any]], telemetry_log_id: int = None) -> Tuple[int, int]:
        """
        Saves the incidents and updates detections with foreign key references in the database.
        Allows exact historical tracking and multi-tier filtering.
        
        Returns:
            Tuple: (num_incidents_inserted, num_detections_inserted)
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        is_pg = self.db_manager.db_type == "postgresql"
        
        incidents_inserted = 0
        detections_inserted = 0
        
        # SQL insert statements
        insert_incident_sql = """
        INSERT INTO incidents (
            centroid_latitude, centroid_longitude, severity, illegal_zone, synced_to_cloud
        ) VALUES (%s, %s, %s, %s, 0) RETURNING id;
        """ if is_pg else """
        INSERT INTO incidents (
            centroid_latitude, centroid_longitude, severity, illegal_zone, synced_to_cloud
        ) VALUES (?, ?, ?, ?, 0);
        """
        
        insert_detection_sql = """
        INSERT INTO detections (
            telemetry_log_id, incident_id, class_name, confidence,
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
            latitude, longitude, frame_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """ if is_pg else """
        INSERT INTO detections (
            telemetry_log_id, incident_id, class_name, confidence,
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
            latitude, longitude, frame_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        try:
            for inc in incidents:
                # 1. Insert Incident
                inc_params = (
                    inc['centroid_lat'],
                    inc['centroid_lon'],
                    inc['severity'],
                    inc['illegal_zone']
                )
                
                cursor.execute(insert_incident_sql, inc_params)
                
                # Fetch generated ID
                if is_pg:
                    incident_id = cursor.fetchone()[0]
                else:
                    incident_id = cursor.lastrowid
                
                incidents_inserted += 1
                
                # 2. Insert Detections linked to this Incident
                for det in inc['detections']:
                    det_params = (
                        telemetry_log_id,
                        incident_id,
                        det['class_name'],
                        det['confidence'],
                        det['bbox_x_min'],
                        det['bbox_y_min'],
                        det['bbox_x_max'],
                        det['bbox_y_max'],
                        det['lat'],
                        det['lon'],
                        det.get('frame_path', '')
                    )
                    cursor.execute(insert_detection_sql, det_params)
                    detections_inserted += 1
                    
            conn.commit()
            logger.info(f"Successfully committed: {incidents_inserted} incidents and {detections_inserted} linked detections to local DB.")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save cluster incidents to database: {e}")
        finally:
            cursor.close()
            conn.close()
            
        return incidents_inserted, detections_inserted

if __name__ == "__main__":
    db = DatabaseManager(db_type="sqlite")
    # Make sure tables exist
    db.initialize_database()
    
    engine = ClusterEngine(db_manager=db)
    
    # Mock some detections near the first flight coordinates
    mock_detections = [
        {'class_name': 'jcb', 'confidence': 0.91, 'bbox_x_min': 500, 'bbox_y_min': 600, 'bbox_x_max': 700, 'bbox_y_max': 800, 'lat': 26.1265, 'lon': 91.6026},
        {'class_name': 'truck', 'confidence': 0.88, 'bbox_x_min': 800, 'bbox_y_min': 600, 'bbox_x_max': 1000, 'bbox_y_max': 800, 'lat': 26.1264, 'lon': 91.6027},
        {'class_name': 'person', 'confidence': 0.85, 'bbox_x_min': 400, 'bbox_y_min': 800, 'bbox_x_max': 450, 'bbox_y_max': 900, 'lat': 26.12645, 'lon': 91.60255},
        # One isolated human far away
        {'class_name': 'person', 'confidence': 0.76, 'bbox_x_min': 100, 'bbox_y_min': 100, 'bbox_x_max': 150, 'bbox_y_max': 150, 'lat': 26.1360, 'lon': 91.6318}
    ]
    
    # Run clustering
    incidents = engine.cluster_detections(mock_detections, eps_meters=50.0)
    print(f"Generated Clusters: {len(incidents)}")
    for inc in incidents:
        print(f"Cluster severity: {inc['severity']}, Illegal: {inc['illegal_zone']}, Detections inside: {len(inc['detections'])}")
        
    # Save to DB
    engine.save_incidents_to_db(incidents, telemetry_log_id=1)
