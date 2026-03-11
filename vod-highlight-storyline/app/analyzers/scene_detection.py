from __future__ import annotations

from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)


def detect_scene_cuts(video_path: Path) -> list[float]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector())
        manager.detect_scenes(video)
        return [scene[0].get_seconds() for scene in manager.get_scene_list()]
    except Exception as exc:
        logger.warning("scene detection failed: %s. Continuing with empty scene cuts.", exc)
        return []
