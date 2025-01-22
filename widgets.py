from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
    QPushButton, QDialog, QRadioButton, QButtonGroup, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from utils import KeypointRenderer, DEFAULT_DISPLAY_SIZE, ORIGINAL_SIZE

class KeypointEditorWidget(QWidget):
    keypoint_updated = pyqtSignal(int, list)  # 키포인트 ID, [x, y]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_variables()
        self._setup_ui()
        
    def _init_variables(self):
        """상태 변수 초기화"""
        self.current_image = None
        self.keypoints = [[0,0] for _ in range(17)]
        self.selected_point = None
        self.dragging = False
        # 2304->1152 (기존 하드코딩된 값을 상수로 대체)
        self.scale_factor = DEFAULT_DISPLAY_SIZE[0] / ORIGINAL_SIZE[0]
        
    def _setup_ui(self):
        """UI 컴포넌트 초기화 및 레이아웃 구성"""
        self.setMinimumSize(1152, 800)
        
        # 메인 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # 이미지 영역 설정
        image_wrapper = self._create_image_area()
        layout.addWidget(image_wrapper)
        
        # 컨트롤 영역 설정
        control_area = self._create_control_area()
        layout.addWidget(control_area)
        
    def _create_image_area(self):
        """이미지 표시 영역 생성"""
        # 이미지 컨테이너
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: black;")
        self.image_container.setFixedSize(1152, 648)
        
        # 이미지 레이블
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        
        # 이미지 래퍼
        image_wrapper = QWidget()
        wrapper_layout = QVBoxLayout(image_wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(self.image_container, alignment=Qt.AlignTop)
        
        return image_wrapper
        
    def _create_control_area(self):
        """컨트롤 영역 생성"""
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(10, 0, 10, 10)
        bottom_layout.setSpacing(5)
        
        # 파일명 표시 레이블
        self.filename_label = QLabel()
        self.filename_label.setStyleSheet("background-color: white; padding: 5px;")
        self.filename_label.setFixedHeight(30)
        bottom_layout.addWidget(self.filename_label)
        
        # 네비게이션 버튼
        buttons_layout = QHBoxLayout()
        self.prev_image_btn = QPushButton("◀ 이전 이미지")
        self.next_image_btn = QPushButton("다음 이미지 ▶")
        buttons_layout.addWidget(self.prev_image_btn)
        buttons_layout.addWidget(self.next_image_btn)
        bottom_layout.addLayout(buttons_layout)
        
        # 단축키 설명
        shortcut_text = (
            "단축키 안내  -  ◀/▶: 이전/다음 이미지  |  "
            "↑/↓: 이전/다음 JSON  |  S: 수동 저장"
        )
        shortcut_label = QLabel(shortcut_text)
        shortcut_label.setStyleSheet("background-color: white; padding: 5px;")
        shortcut_label.setFixedHeight(30)
        bottom_layout.addWidget(shortcut_label)
        
        return bottom_container
        
    def mousePressEvent(self, event):
        if self.current_image is None or len(self.keypoints) == 0:
            return

        # 마우스 좌표를 원본 이미지 좌표로 변환
        x = event.pos().x() / self.scale_factor
        y = event.pos().y() / self.scale_factor

        # 기존 키포인트 선택 확인
        for i, point in enumerate(self.keypoints):
            if point[0] == 0 and point[1] == 0:
                continue

            # 스케일링된 좌표로 거리 계산
            scaled_x = point[0] * self.scale_factor
            scaled_y = point[1] * self.scale_factor
            dx = scaled_x - event.pos().x()
            dy = scaled_y - event.pos().y()

            if (dx*dx + dy*dy) <= 100:  # 선택 반경
                self.selected_point = i
                self.dragging = True
                self.update_view()
                return

        self.selected_point = None
        self.dragging = False
        self.update_view()

    def mouseMoveEvent(self, event):
        if not self.dragging or self.selected_point is None:
            return

        # UI 좌표를 원본 이미지 좌표로 변환
        x = event.pos().x() / self.scale_factor
        y = event.pos().y() / self.scale_factor

        # 선택된 키포인트 위치 업데이트
        self.keypoints[self.selected_point] = [x, y]
        self.keypoint_updated.emit(self.selected_point, [x, y])
        self.update_view()
            
    def mouseDoubleClickEvent(self, event):
        if self.current_image is None:
            return

        x = event.pos().x() / self.scale_factor
        y = event.pos().y() / self.scale_factor

        # 기존 키포인트 삭제 처리
        for i, point in enumerate(self.keypoints):
            if point[0] == 0 and point[1] == 0:  # 비활성화된 점은 무시
                continue

            dx = point[0] - x
            dy = point[1] - y
            if (dx * dx + dy * dy) <= 100:  # 선택 반경 내에 있는 경우
                reply = QMessageBox.question(
                    self, '키포인트 삭제',
                    f'{i + 1}번 키포인트를 삭제하시겠습니까?',
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.keypoints[i] = [0, 0]
                    self.keypoint_updated.emit(i, [0, 0])
                    self.update_view()
                return

        # 새 키포인트 추가 다이얼로그
        dialog = KeypointDialog(self.keypoints, self)
        if dialog.exec_():
            point_id = dialog.selected_point
            if point_id is not None:
                self.keypoints[point_id] = [x, y]
                self.keypoint_updated.emit(point_id, [x, y])
                self.update_view()

    def update_view(self):
        if self.current_image is None:
            return

        # 키포인트 렌더링
        rendered = KeypointRenderer.render_skeleton(
            self.current_image,
            self.keypoints,
            self.selected_point  # 선택된 키포인트 강조
        )

        # 크기 조정 및 QImage 변환
        resized = cv2.resize(rendered, (1152, 648))  # UI 크기에 맞게 조정
        h, w, c = resized.shape
        bytes_per_line = 3 * w
        qimg = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        
class KeypointDialog(QDialog):
    def __init__(self, existing_points, parent=None):
        super().__init__(parent)
        self.setWindowTitle('키포인트 추가')
        self.selected_point = None

        # 기존 포인트 리스트 길이 보정 (최소 13개 유지)
        if len(existing_points) < 13:
            existing_points.extend([[0, 0]] * (13 - len(existing_points)))

        layout = QVBoxLayout()

        # 키포인트 버튼 그룹
        self.point_group = QButtonGroup()
        point_names = [
            "1: 코", "2: 오른쪽 어깨", "3: 왼쪽 어깨",
            "4: 오른쪽 팔꿈치", "5: 왼쪽 팔꿈치",
            "6: 오른쪽 손목", "7: 왼쪽 손목",
            "8: 오른쪽 골반", "9: 왼쪽 골반",
            "10: 오른쪽 무릎", "11: 왼쪽 무릎",
            "12: 오른쪽 발목", "13: 왼쪽 발목"
        ]

        for i, name in enumerate(point_names):
            radio = QRadioButton(name)
            self.point_group.addButton(radio, i)
            layout.addWidget(radio)

            # 이미 사용된 점은 비활성화 및 회색으로 표시
            if existing_points[i] != [0, 0]:
                radio.setEnabled(False)
                radio.setStyleSheet("color: #808080;")  # 회색으로 표시

        # 버튼
        buttons = QHBoxLayout()
        ok_btn = QPushButton('확인')
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton('취소')
        cancel_btn.clicked.connect(self.reject)

        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        self.setLayout(layout)
        
    def accept(self):
        self.selected_point = self.point_group.checkedId()
        if self.selected_point is None:
            QMessageBox.warning(self, '경고', '키포인트를 선택해주세요.')
            return
        super().accept()
        
def handle_click_event(keypoints, clicked_index):
    """
    클릭된 키포인트의 좌표를 수정합니다.
    
    Args:
        keypoints (list): 현재 키포인트 리스트.
        clicked_index (int): 클릭된 키포인트의 인덱스.

    Returns:
        None
    """
    # 클릭된 키포인트의 좌표를 (0, 0)으로 설정
    if clicked_index < len(keypoints):
        keypoints[clicked_index] = [0, 0]
