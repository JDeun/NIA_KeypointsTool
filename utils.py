from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import logging
import json
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 키포인트/이미지 관련 상수
DEFAULT_DISPLAY_SIZE = (1152, 648)
ORIGINAL_SIZE = (2304, 1296)

COLORS = {
    'red': (0, 0, 255),      # BGR
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'white': (255, 255, 255),
    'yellow': (0, 255, 255),  # BGR
    'grey' : (128, 128, 128)
}
    
CONNECTIONS = [
    (1, 6), (1, 7), (6, 7),   # 코-어깨 삼각형
    (6, 8), (8, 10),          # 오른쪽: 어깨 -> 팔꿈치 -> 손목
    (7, 9), (9, 11),          # 왼쪽: 어깨 -> 팔꿈치 -> 손목
    (6, 12), (12, 14), (14, 16),  # 오른쪽: 어깨 -> 골반 -> 무릎 -> 발목
    (7, 13), (13, 15), (15, 17),   # 왼쪽: 어깨 -> 골반 -> 무릎 -> 발목
    (12, 13)  # 골반 연결
]

# KeypointRenderer 최적화
class KeypointRenderer:
    @staticmethod
    def render_skeleton(image, keypoints, selected_point=None):
        rendered = image.copy()
        h, w = rendered.shape[:2]
        scale_x = w / ORIGINAL_SIZE[0]
        scale_y = h / ORIGINAL_SIZE[1]
        
        # 좌표 변환을 미리 계산
        scaled_keypoints = [
            (int(kp[0] * scale_x), int(kp[1] * scale_y))
            for kp in keypoints
        ]
        
        # 연결선 일괄 처리
        for start_idx, end_idx in CONNECTIONS:
            start_point = scaled_keypoints[start_idx-1]
            end_point = scaled_keypoints[end_idx-1]
            
            if all(p != (0, 0) for p in (start_point, end_point)):
                cv2.line(rendered, start_point, end_point, COLORS['white'], 2)
        
        # 키포인트 렌더링 최적화
        for idx, (x, y) in enumerate(scaled_keypoints):
            if (x, y) == (0, 0):
                continue
                
            actual_idx = idx + 1
            color = KeypointRenderer.get_point_color(idx)
            
            if selected_point == actual_idx:
                cv2.circle(rendered, (x, y), 6, COLORS['white'], -1)
            cv2.circle(rendered, (x, y), 5, color, -1)
            cv2.putText(rendered, str(actual_idx), (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        return rendered
    
    @staticmethod
    def get_point_color(index):
        """
        키포인트 색상 매핑: 
        1(nose) → yellow
        신체 우측 → red
        신체 좌측 → green
        """
        if index == 0:  # nose
            return COLORS['yellow']  # yellow
        elif index in [5, 7, 9, 11]:  # 신체 우측
            return COLORS['red']  # red
        elif index in [6, 8, 10, 12]:  # 신체 좌측
            return COLORS['green']  # green
        else:
            return COLORS['grey']  # grey (기본값)

# ImageCache 클래스 최적화
class ImageCache:
    def __init__(self):
        self.cache = OrderedDict()
        self.max_size = self._calculate_max_cache_size()
        
    def _calculate_max_cache_size(self):
        """시스템 메모리 기반으로 최적의 캐시 크기 계산"""
        available_memory = psutil.virtual_memory().available
        image_size = ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1] * 3  # RGB
        max_images = int(available_memory * 0.25 / image_size)
        return min(30, max_images)
        
    def put(self, path, image):
        if len(self.cache) >= self.max_size:
            # LRU 방식으로 오래된 항목 제거
            self.cache.popitem(last=False)
        self.cache[path] = image
        
    def get(self, path):
        if path in self.cache:
            # 캐시 히트 시 항목을 최신 위치로 이동
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        return None

    def clear(self):
        self.cache.clear()

def get_json_path(image_path: Path, check_edited: bool = True) -> Path:
    """이미지 파일에 대응하는 JSON 파일 경로 반환"""
    base_path = image_path.parent.parent.parent  # 상위 폴더로 이동
    json_folder = base_path / "2.라벨링데이터" / image_path.parent.name
    
    if check_edited:
        edited_path = json_folder / "edited" / f"{image_path.stem}.json"
        if edited_path.exists():
            return edited_path
            
    return json_folder / f"{image_path.stem}.json"

def scale_keypoints_to_image(keypoints, image_width, image_height, original_width=2304, original_height=1296):
    """
    키포인트 좌표를 실제 이미지 크기에 맞게 스케일링합니다.
    :param keypoints: 원본 키포인트 좌표 리스트.
    :param image_width: 실제 이미지의 너비.
    :param image_height: 실제 이미지의 높이.
    :return: 스케일링된 키포인트 리스트.
    """
    scaled_keypoints = []
    for x, y in keypoints:
        scaled_x = int(x / original_width * image_width)
        scaled_y = int(y / original_height * image_height)
        scaled_keypoints.append([scaled_x, scaled_y])
    return scaled_keypoints

