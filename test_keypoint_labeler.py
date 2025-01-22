# test_keypoint_labeler.py

import pytest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from pathlib import Path
import numpy as np
import cv2
import json
from unittest.mock import MagicMock, patch

from main import KeypointLabeler
from widgets import KeypointEditorWidget, KeypointDialog
from utils import KeypointRenderer, ImageCache

# Fixtures
@pytest.fixture
def app(qtbot):
    app = KeypointLabeler()
    qtbot.addWidget(app)
    return app

@pytest.fixture
def editor(qtbot):
    widget = KeypointEditorWidget()
    qtbot.addWidget(widget)
    return widget

@pytest.fixture
def sample_image():
    """테스트용 더미 이미지"""
    return np.zeros((1296, 2304, 3), dtype=np.uint8)

@pytest.fixture
def sample_keypoints():
    """17개 키포인트용 더미 데이터"""
    return [[100, 100] for _ in range(17)]

# 단위 테스트: KeypointEditorWidget
class TestKeypointEditorWidget:
    def test_init(self, editor):
        """기본 초기화 테스트"""
        assert editor.current_image is None
        assert len(editor.keypoints) == 17  # 17개 키포인트
        assert editor.selected_point is None
        assert not editor.dragging
        assert abs(editor.scale_factor - 0.5) < 1e-6

    @patch('widgets.KeypointRenderer')
    def test_mouse_interaction(self, mock_renderer, qtbot, editor, sample_image):
        """마우스 상호작용 테스트"""
        editor.current_image = sample_image
        mock_renderer.render_skeleton.return_value = sample_image
        
        # 클릭 테스트 - 빈 영역
        qtbot.mouseClick(editor, Qt.LeftButton, pos=QPoint(100, 100))
        assert editor.selected_point is None
        
        # 드래그 테스트 - 키포인트가 있는 위치
        editor.keypoints[6] = [100, 100]  # 오른쪽 어깨
        qtbot.mousePress(editor, Qt.LeftButton, pos=QPoint(50, 50))
        qtbot.mouseMove(editor, QPoint(60, 60))
        qtbot.mouseRelease(editor, Qt.LeftButton)

    @patch('widgets.KeypointDialog')
    @patch('widgets.KeypointRenderer')
    def test_double_click(self, mock_renderer, mock_dialog, qtbot, editor, sample_image):
        """더블클릭으로 키포인트 추가/삭제 테스트"""
        editor.current_image = sample_image
        mock_renderer.render_skeleton.return_value = sample_image
        mock_dialog.return_value.exec_.return_value = True
        mock_dialog.return_value.selected_point = 6  # 오른쪽 어깨
        
        # 더블클릭으로 키포인트 추가
        qtbot.mouseDClick(editor, Qt.LeftButton, pos=QPoint(100, 100))

# 단위 테스트: KeypointRenderer
def test_renderer(sample_image, sample_keypoints):
    """KeypointRenderer 테스트"""
    rendered = KeypointRenderer.render_skeleton(sample_image, sample_keypoints)
    assert rendered.shape == sample_image.shape
    assert isinstance(rendered, np.ndarray)

# 단위 테스트: ImageCache
class TestImageCache:
    def test_cache_operations(self, sample_image):
        cache = ImageCache()
        
        # Put & Get
        cache.put("test.jpg", sample_image)
        assert np.array_equal(cache.get("test.jpg"), sample_image)
        
        # Cache size limit
        for i in range(35):  # max_size(30) 이상
            cache.put(f"test_{i}.jpg", sample_image)
        assert len(cache.cache) <= 30

# 통합 테스트
class TestKeypointLabeler:
    @patch.object(QFileDialog, 'getExistingDirectory')
    def test_folder_selection(self, mock_dialog, app, qtbot, tmp_path):
        """폴더 선택 기능 테스트"""
        test_dir = tmp_path / "test_data"
        mock_dialog.return_value = str(test_dir)
        
        app.select_folder()
        assert Path(app.path_label.text()) == test_dir

    @patch('main.KeypointLabeler.load_folder_files')
    @patch('cv2.imdecode')
    def test_json_loading(self, mock_cv2, mock_load, app, qtbot, tmp_path):
        """JSON 파일 로드 테스트"""
        test_dir = tmp_path / "test_data"
        image_dir = test_dir / "1.추출 이미지 데이터" / "test"
        json_dir = test_dir / "2.라벨링데이터" / "test"
        image_dir.mkdir(parents=True)
        json_dir.mkdir(parents=True)
        
        # 이미지 파일 생성
        mock_cv2.return_value = np.zeros((1296, 2304, 3), dtype=np.uint8)
        test_image = image_dir / "test_0.jpg"
        with open(test_image, 'wb') as f:
            f.write(b'dummy image data')
            
        # JSON 파일 생성
        test_json = json_dir / "test.json"
        test_data = {
            "segmentation": [{
                "keyframes": "0",
                "keypoints": [[100,100] for _ in range(17)]
            }]
        }
        with open(test_json, 'w') as f:
            json.dump(test_data, f)
            
        # base_path 설정 및 테스트
        app.base_path = test_dir
        app.load_json(test_json)
        assert app.current_json == test_json

    @patch('PyQt5.QtWidgets.QMessageBox.critical')
    def test_error_handling(self, mock_critical, app, qtbot):
        """에러 처리 테스트"""
        app.load_json(Path("nonexistent.json"))
        mock_critical.assert_called_once()

    # test_save_functionality 수정
    @patch('PyQt5.QtWidgets.QMessageBox.question')  # QMessageBox 클래스가 아닌 question 메소드만 패치
    def test_save_functionality(self, mock_question, app, qtbot, tmp_path):
        """저장 기능 테스트"""
        test_dir = tmp_path / "test_data"
        json_dir = test_dir / "2.라벨링데이터" / "test"
        json_dir.mkdir(parents=True)
        
        # JSON 파일 생성
        test_json = json_dir / "test.json"
        test_data = {"keypoints": {}}
        with open(test_json, 'w') as f:
            json.dump(test_data, f)
            
        # 저장 테스트 설정
        app.base_path = test_dir
        app.current_json = test_json
        app.modified = True
        
        # 저장 대화상자 응답 설정
        mock_question.return_value = QMessageBox.Yes
        
        app.save_check()
        assert mock_question.called  # question 메소드가 호출되었는지 확인