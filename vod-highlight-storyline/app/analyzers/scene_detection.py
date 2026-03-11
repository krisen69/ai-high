from __future__ import annotations

from pathlib import Path


def detect_scene_cuts(video_path: Path) -> list[float]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector())
        manager.detect_scenes(video)
        return [s[0].get_seconds() for s in manager.get_scene_list()]
    except Exception:
        return []
