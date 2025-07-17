from typing import Dict
from datetime import datetime
import logging

class LivenessCheckFailure(Exception):
    """Custom exception for liveness check failures"""
    pass

def handle_liveness_failure(detection_result: Dict) -> None:
    """
    Handle cases where face liveness check fails due to potential spoofing
    
    Args:
        detection_result: Dictionary containing liveness check results
    Raises:
        LivenessCheckFailure: When spoofing is detected
    """
    logger = logging.getLogger(__name__)
    
    logger.warning(
        "Face liveness check failed at %s. Potential spoofing detected.",
        datetime.now().isoformat()
    )
    
    
    raise LivenessCheckFailure("Face liveness check failed: Potential spoofing detected")

def verify_liveness(face_data: Dict) -> bool:
    if face_data.get('liveness_score', 0) < 0.9:
        handle_liveness_failure(face_data)
    return True