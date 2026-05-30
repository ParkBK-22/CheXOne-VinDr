from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QAExample:
    qa_id: str
    image_id: str
    image_path: str
    question_type: str
    question: str
    answer: str
    finding: Optional[str] = None
    bbox: Optional[List[float]] = None
    split: str = "test"
    metadata: Optional[Dict[str, Any]] = None
