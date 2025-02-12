import sys
from pathlib import Path
import json
import cv2
import numpy as np
import logging
import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QProgressDialog,
    QPushButton, QFileDialog, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from widgets import KeypointEditorWidget
from utils import ImageCache, get_json_path, KeypointRenderer

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeypointLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('키포인트 라벨링 도구')
        self.resize(1280, 720)
        
        # 상태 변수 초기화
        self.base_path = None
        self.current_json = None
        self.current_images = []  # 현재 JSON에 속한 이미지들
        self.current_image_idx = -1
        self.modified = False
        self.image_cache = ImageCache()
        self.keypoints_data = {}  # 키프레임별 키포인트 데이터 저장
                
        # UI 초기화
        self.init_ui()
        self.setup_shortcuts()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # 좌측: 이미지 편집 영역
        left_layout = self.init_left_section()
        
        # 우측: 파일 목록 영역을 담을 컨테이너 위젯
        right_container = QWidget()
        right_container.setMinimumWidth(300)
        right_layout = self.init_right_section()
        right_container.setLayout(right_layout)
        
        layout.addLayout(left_layout, stretch=75)
        layout.addWidget(right_container, stretch=25)
        
        main_widget.setLayout(layout)

    def init_right_section(self):
        layout = QVBoxLayout()
        
        # 상위 폴더 선택
        folder_layout = QHBoxLayout()
        select_btn = QPushButton("폴더 선택")
        select_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(select_btn)
        
        self.path_label = QLabel()
        folder_layout.addWidget(self.path_label)
        layout.addLayout(folder_layout)
        
        # 하위 폴더 선택
        folder_combo_layout = QHBoxLayout()
        folder_combo_layout.addWidget(QLabel("폴더:"))
        self.folder_combo = QComboBox()
        self.folder_combo.currentIndexChanged.connect(self.load_folder_files)
        folder_combo_layout.addWidget(self.folder_combo)
        layout.addLayout(folder_combo_layout)
        
        # 파일 목록
        self.file_list = QTableWidget()
        self.file_list.setColumnCount(3)
        self.file_list.setHorizontalHeaderLabels(['파일명', '상태', '동작'])
        header = self.file_list.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.file_list)
        
        # 저장 버튼
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_current)
        layout.addWidget(self.save_btn)
        
        return layout
        
    def init_left_section(self):
        layout = QVBoxLayout()
        
        # 키포인트 에디터 위젯
        self.editor_widget = KeypointEditorWidget()
        self.editor_widget.keypoint_updated.connect(self.on_keypoint_update)
        layout.addWidget(self.editor_widget)
        
        return layout

    def setup_shortcuts(self):
        """단축키 설정"""
        self.setFocusPolicy(Qt.StrongFocus)
        
    def keyPressEvent(self, event):
        """키보드 단축키 처리"""
        if event.key() == Qt.Key_Left:  # 이전 이미지
            self.move_prev_image()
        elif event.key() == Qt.Key_Right:  # 다음 이미지
            self.move_next_image()
        elif event.key() == Qt.Key_Up:  # 이전 JSON
            self.move_prev_json()
        elif event.key() == Qt.Key_Down:  # 다음 JSON
            self.move_next_json()
        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:  # 저장
            self.save_current()
            
    def select_folder(self):
        """상위 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            self.base_path = Path(folder)
            self.path_label.setText(folder)
            
            # 하위 폴더 목록 업데이트
            if (self.base_path / "1.추출 이미지 데이터").exists():
                folders = [d.name for d in (self.base_path / "1.추출 이미지 데이터").iterdir() 
                        if d.is_dir()]
                self.folder_combo.clear()
                self.folder_combo.addItems(sorted(folders))
                
    def load_folder_files(self):
        """하위 폴더 내 JSON 파일 목록 로드"""
        try:
            if not self.base_path or not self.folder_combo.currentText():
                return
                
            folder_name = self.folder_combo.currentText()
            json_folder = self.base_path / "2.라벨링데이터" / folder_name
            edited_folder = json_folder / "edited"
            
            # 파일 목록 가져오기
            json_files = sorted(list(json_folder.glob("*.json")))
            
            # 프로그레스 다이얼로그 설정
            progress = QProgressDialog("파일 목록 로딩 중...", None, 0, len(json_files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)  # 즉시 표시
            progress.setCancelButton(None)   # 취소 버튼 제거
            
            # UI 업데이트 중단
            self.file_list.setUpdatesEnabled(False)
            self.file_list.setSortingEnabled(False)
            self.file_list.setRowCount(0)
            
            # 전체 파일 로드
            for i, json_file in enumerate(json_files):
                row = self.file_list.rowCount()
                self.file_list.insertRow(row)
                
                # 파일명
                name_item = QTableWidgetItem(json_file.name)
                self.file_list.setItem(row, 0, name_item)
                
                # 상태
                status = "수정됨" if (edited_folder / json_file.name).exists() else "수정 사항 없음"
                status_item = QTableWidgetItem(status)
                self.file_list.setItem(row, 1, status_item)
                
                # 로드 버튼
                load_btn = QPushButton("로드")
                load_btn.clicked.connect(lambda checked, f=json_file: self.load_json(f))
                self.file_list.setCellWidget(row, 2, load_btn)
                
                # 프로그레스 업데이트
                progress.setValue(i + 1)
                
                # 50개 파일마다 이벤트 처리 (UI 반응성 유지)
                if i % 50 == 0:
                    QApplication.processEvents()
            
            # UI 업데이트 재개
            self.file_list.setUpdatesEnabled(True)
            self.file_list.setSortingEnabled(True)
            progress.close()
                
        except Exception as e:
            logger.error(f"파일 목록 로드 실패: {e}")
            QMessageBox.critical(self, "오류", f"파일 목록 로드 실패: {e}")

    def load_json(self, json_file: Path):
        try:
            if self.modified:
                self.save_check()

            # edited 폴더의 파일을 우선적으로 확인
            edited_json = json_file.parent / "edited" / json_file.name
            load_path = edited_json if edited_json.exists() else json_file

            logger.info(f"JSON 파일 로드 시도: {load_path}")

            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.keypoints_data = {}

            # segmentation 데이터 처리
            if 'segmentation' in data:
                for segment in data['segmentation']:
                    frame_num = segment.get('keyframe')  # int 형태로 유지
                    keypoints = segment.get('keypoints', [])
                    if keypoints:  # 빈 데이터 체크
                        # 좌표값을 정수형으로 보장
                        processed_keypoints = [[int(x), int(y)] for x, y in keypoints]
                        self.keypoints_data[frame_num] = processed_keypoints
                        logger.info(f"프레임 {frame_num}의 키포인트 데이터 로드: {processed_keypoints}")

            # 관련 이미지 파일 찾기
            image_folder = self.base_path / "1.추출 이미지 데이터" / json_file.parent.name
            prefix = json_file.stem
            self.current_images = sorted(image_folder.glob(f"{prefix}_*.jpg"))

            if not self.current_images:
                raise FileNotFoundError(f"이미지 파일이 없습니다: {image_folder}")

            logger.info(f"이미지 파일 찾음: {len(self.current_images)}개")

            self.current_json = json_file
            self.current_image_idx = 0
            self.modified = False  # edited 파일을 로드했더라도 초기에는 수정되지 않은 상태로 설정

            self.load_image(self.current_images[0])
            self.update_file_list()

        except Exception as e:
            logger.error(f"JSON 로드 실패: {str(e)}")
            QMessageBox.critical(self, "오류", f"JSON 로드 실패: {str(e)}")

    # 키포인트 필터링 함수
    def filter_keypoints(self, raw_keypoints):
        """
        키포인트 데이터를 필터링하여 유효한 좌표만 반환.
        :param raw_keypoints: [[x, y], ...] 형식의 데이터
        :return: [(index, (x, y)), ...] 형식의 필터링된 데이터
        """
        filtered = []
        try:
            for index, coords in enumerate(raw_keypoints):
                if len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords):
                    x, y = map(int, coords)
                    if (x, y) != (0, 0):
                        filtered.append((index, (x, y)))
                    else:
                        logger.warning(f"키포인트 {index}가 (0, 0) 상태입니다: {coords}")
                else:
                    logger.warning(f"키포인트 {index}의 형식이 잘못되었습니다: {coords}")
        except Exception as e:
            logger.error(f"키포인트 필터링 중 오류 발생: {e}")
        return filtered


    def load_image(self, image_path: Path):
        """이미지 및 해당 키포인트 데이터 로드"""
        try:
            # 이미지 캐시 체크 및 읽기
            image = self.image_cache.get(str(image_path))
            if image is None:
                image = cv2.imdecode(
                    np.fromfile(image_path.as_posix(), np.uint8),
                    cv2.IMREAD_COLOR
                )
                if image is None:
                    raise Exception("이미지를 읽을 수 없습니다.")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.image_cache.put(str(image_path), image)

            # 키프레임 번호 추출 - int로 변환
            keyframe_num = int(image_path.stem.split('_')[-1])
            logger.info(f"키프레임 번호: {keyframe_num}")

            # 키포인트 데이터 로드
            if keyframe_num in self.keypoints_data:
                keypoints = self.keypoints_data[keyframe_num]
                logger.info(f"키포인트 데이터 찾음: {keypoints}")
            else:
                # 17개 포인트 초기화 (1개는 코, 4개는 눈/귀, 12개는 신체 포인트)
                keypoints = [[0,0] for _ in range(17)]
                logger.info("키포인트 데이터 없음, 기본값 사용")

            # 에디터 위젯 업데이트
            self.editor_widget.current_image = image
            self.editor_widget.keypoints = keypoints
            self.editor_widget.selected_point = None  # 선택 초기화
            self.editor_widget.update_view()
            self.editor_widget.filename_label.setText(image_path.name)

        except Exception as e:
            logger.error(f"이미지 로드 실패: {str(e)}")
            QMessageBox.critical(self, "오류", f"이미지 로드 실패: {str(e)}")

    def on_keypoint_update(self, point_id: int, coords: list):
        """
        키포인트 업데이트 메서드
        
        Args:
            point_id (int): 키포인트 ID
            coords (list): [x, y] 좌표
        """
        try:
            current_image = self.current_images[self.current_image_idx]
            keyframe_num = int(current_image.stem.split('_')[-1])
            
            # keyframe_num을 int로 유지하고 17개 포인트로 초기화
            if keyframe_num not in self.keypoints_data:
                self.keypoints_data[keyframe_num] = [[0,0]] * 17
            
            # 좌표를 정수형으로 변환하여 저장
            x, y = coords
            self.keypoints_data[keyframe_num][point_id] = [int(x), int(y)]
            logger.info(f"키포인트 업데이트: 프레임 {keyframe_num}, 포인트 {point_id}, 좌표 {[int(x), int(y)]}")
            
            self.modified = True
            self.update_file_list()
            
        except Exception as e:
            logger.error(f"키포인트 업데이트 실패: {str(e)}")

    def save_current(self):
        try:
            if not self.current_json or not self.modified:
                return
                
            edited_folder = self.current_json.parent / "edited"
            edited_folder.mkdir(exist_ok=True)
            save_path = edited_folder / self.current_json.name
            
            # 현재 JSON 데이터 로드
            with open(self.current_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 현재 이미지의 키프레임 번호 추출
            current_image = self.current_images[self.current_image_idx]
            keyframe_num = int(current_image.stem.split('_')[-1])
            
            # segmentation 배열에서 해당 키프레임 데이터 업데이트
            for segment in data.get('segmentation', []):
                if segment.get('keyframe') == keyframe_num:  # 'keyframe'으로 수정
                    segment['keypoints'] = self.keypoints_data[keyframe_num]  # editor_widget 대신 저장된 데이터 사용
                    break
            
            # 저장
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.modified = False
            self.update_file_list()
            
            # 저장 완료 메시지
            msg = QMessageBox(self)
            msg.setText("수정사항이 저장되었습니다.")
            msg.setWindowTitle("알림")
            QTimer.singleShot(1000, msg.close)
            msg.show()
            
        except Exception as e:
            logger.error(f"저장 실패: {e}")
            QMessageBox.critical(self, "오류", f"저장 실패: {e}")

    def save_check(self):
        """수정사항 있을 경우 저장 확인"""
        if self.modified:
            reply = QMessageBox.question(
                self, '확인',
                '저장되지 않은 변경사항이 있습니다. 저장하시겠습니까?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.save_current()

    def update_file_list(self):
        """파일 목록 상태 업데이트"""
        for row in range(self.file_list.rowCount()):
            item = self.file_list.item(row, 0)
            # 모든 행의 배경색을 먼저 흰색으로 초기화
            for col in range(3):
                if self.file_list.item(row, col):
                    self.file_list.item(row, col).setBackground(QColor("white"))
            
            if item:
                file_path = self.current_json.parent / item.text()
                edited_path = file_path.parent / "edited" / item.text()
                
                # 현재 선택된 파일 하이라이트
                if self.current_json and item.text() == self.current_json.name:
                    for col in range(3):
                        if self.file_list.item(row, col):
                            self.file_list.item(row, col).setBackground(QColor("#E3F2FD"))
                    # 현재 선택된 파일의 상태는 수정 중/수정됨 구분
                    if self.modified:
                        status = "수정 중"
                    else:
                        status = "수정됨" if edited_path.exists() else "수정 사항 없음"
                else:
                    # 다른 파일들은 edited 폴더 존재 여부만 확인
                    status = "수정됨" if edited_path.exists() else "수정 사항 없음"
                
                self.file_list.item(row, 1).setText(status)

    def closeEvent(self, event):
        """프로그램 종료"""
        if self.modified:
            reply = QMessageBox.question(
                self, '확인',
                '저장되지 않은 변경사항이 있습니다. 저장하시겠습니까?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                self.save_current()
        event.accept()
        
    def move_next_image(self):
        """다음 이미지로 이동"""
        try:
            if not self.current_images:
                return
                
            if self.current_image_idx < len(self.current_images) - 1:
                self.current_image_idx += 1
                self.load_image(self.current_images[self.current_image_idx])
            else:
                # 마지막 이미지에서 다음 JSON으로
                next_json = self.get_next_json()
                if next_json:
                    self.load_json(next_json)
                    
        except Exception as e:
            logger.error(f"다음 이미지 이동 실패: {e}")
            QMessageBox.critical(self, "오류", f"다음 이미지 이동 실패: {e}")

    def move_prev_image(self):
        """이전 이미지로 이동"""
        try:
            if not self.current_images:
                return
                
            if self.current_image_idx > 0:
                self.current_image_idx -= 1
                self.load_image(self.current_images[self.current_image_idx])
            else:
                # 첫 이미지에서 이전 JSON의 마지막 이미지로
                prev_json = self.get_prev_json()
                if prev_json:
                    self.load_json(prev_json)
                    self.current_image_idx = len(self.current_images) - 1
                    self.load_image(self.current_images[self.current_image_idx])
                    
        except Exception as e:
            logger.error(f"이전 이미지 이동 실패: {e}")
            QMessageBox.critical(self, "오류", f"이전 이미지 이동 실패: {e}")

    def move_next_json(self):
        """다음 JSON으로 이동"""
        next_json = self.get_next_json()
        if next_json:
            self.load_json(next_json)

    def move_prev_json(self):
        """이전 JSON으로 이동"""
        prev_json = self.get_prev_json()
        if prev_json:
            self.load_json(prev_json)

    def get_next_json(self) -> Path:
        """다음 JSON 파일 경로 반환"""
        try:
            current_row = -1
            for row in range(self.file_list.rowCount()):
                if self.file_list.item(row, 0).text() == self.current_json.name:
                    current_row = row
                    break
                    
            if current_row != -1 and current_row < self.file_list.rowCount() - 1:
                next_name = self.file_list.item(current_row + 1, 0).text()
                return self.current_json.parent / next_name
                
        except Exception as e:
            logger.error(f"다음 JSON 파일 찾기 실패: {e}")
        return None

    def get_prev_json(self) -> Path:
        """이전 JSON 파일 경로 반환"""
        try:
            current_row = -1
            for row in range(self.file_list.rowCount()):
                if self.file_list.item(row, 0).text() == self.current_json.name:
                    current_row = row
                    break
                    
            if current_row > 0:
                prev_name = self.file_list.item(current_row - 1, 0).text()
                return self.current_json.parent / prev_name
                
        except Exception as e:
            logger.error(f"이전 JSON 파일 찾기 실패: {e}")
        return None

    def get_keypoints_for_image(self, image_path: Path) -> list:
        """특정 이미지의 키포인트 데이터 반환"""
        try:
            keyframe_num = int(image_path.stem.split('_')[-1])
            json_path = self.current_json
            if (self.current_json.parent / "edited" / self.current_json.name).exists():
                json_path = self.current_json.parent / "edited" / self.current_json.name
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('keypoints', {}).get(str(keyframe_num), [[0,0]]*13)
                
        except Exception as e:
            logger.error(f"키포인트 데이터 로드 실패: {e}")
            return [[0,0]] * 13

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = KeypointLabeler()
    window.show()
    sys.exit(app.exec_())
