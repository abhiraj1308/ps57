import logging
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from datetime import datetime, timezone

from src.models import MissionMetadata, SurveyReport, Detection, SonarImageInfo, DetectionSummary
from src.exceptions import PipelineError
from src.pipeline.geotagging_engine import GeotaggingEngine

def _process_single_file(
    sonar_file: str,
    detections: List[Detection],
    image_info: SonarImageInfo,
    mission_metadata: MissionMetadata,
    config_path: str,
    sonar_system: str
) -> SurveyReport:
    """Worker function to process a single file."""
    engine = GeotaggingEngine(config_path=config_path, sonar_system=sonar_system)
    try:
        engine.load_sonar_data(sonar_file)
        engine.process_detections(detections, image_info, merge_overlapping=True)
        return engine.generate_report(mission_metadata)
    except Exception as e:
        raise PipelineError(detail=f"Error processing {sonar_file}: {e}", stage="process_file")


class BatchProcessor:
    """
    Processes multiple sonar files in parallel.
    Uses ProcessPoolExecutor for CPU-bound work.
    """
    
    def __init__(self, config_path: str = 'config.yaml', max_workers: int = 4, sonar_system: str = 'klein_3000'):
        self.config_path = config_path
        self.max_workers = max_workers
        self.sonar_system = sonar_system
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results: Dict[str, SurveyReport] = {}
        self.progress: Dict[str, str] = {}  # filepath -> status
    
    def process_batch(
        self,
        sonar_files: List[str],
        detections_map: Dict[str, List[Detection]],  # filepath -> detections
        image_info_map: Dict[str, SonarImageInfo],  # filepath -> image info
        mission_metadata: MissionMetadata,
        progress_callback: Optional[Callable[[str, str, float], None]] = None
    ) -> Dict[str, SurveyReport]:
        """
        Process multiple sonar files in parallel.
        """
        total = len(sonar_files)
        completed = 0
        
        for f in sonar_files:
            self.progress[f] = "Pending"
            
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for file in sonar_files:
                self.progress[file] = "Processing"
                if progress_callback:
                    progress_callback(file, "Processing", completed / total)
                
                dets = detections_map.get(file, [])
                info = image_info_map.get(file)
                if not info:
                    self.logger.warning(f"No image info for {file}, skipping")
                    self.progress[file] = "Failed (No image info)"
                    completed += 1
                    continue
                    
                future = executor.submit(
                    _process_single_file,
                    file, dets, info, mission_metadata, self.config_path, self.sonar_system
                )
                futures[future] = file
                
            for future in as_completed(futures):
                file = futures[future]
                try:
                    report = future.result()
                    self.results[file] = report
                    self.progress[file] = "Completed"
                except Exception as e:
                    self.logger.error(f"Failed to process {file}: {e}")
                    self.progress[file] = f"Failed ({str(e)})"
                    
                completed += 1
                if progress_callback:
                    progress_callback(file, self.progress[file], completed / total)
                    
        return self.results
    
    def get_progress(self) -> Dict[str, str]:
        """Get current processing status for all files."""
        return self.progress
    
    def aggregate_reports(self, reports: Dict[str, SurveyReport], mission_metadata: MissionMetadata) -> SurveyReport:
        """
        Merge multiple per-file reports into a single unified report.
        Combines all detections, recalculates summary stats.
        """
        all_detections = []
        for r in reports.values():
            all_detections.extend(r.detections)
            
        summary = DetectionSummary(
            total_detections=len(all_detections),
            detections_by_class={},
            avg_confidence=0.0,
            total_pings_processed=0,
        )
        
        if all_detections:
            total_conf = 0.0
            for d in all_detections:
                summary.detections_by_class[d.classification] = summary.detections_by_class.get(d.classification, 0) + 1
                total_conf += d.confidence_score
            summary.avg_confidence = total_conf / len(all_detections)
            
        return SurveyReport(
            mission_metadata=mission_metadata,
            detections=all_detections,
            summary=summary,
        )
