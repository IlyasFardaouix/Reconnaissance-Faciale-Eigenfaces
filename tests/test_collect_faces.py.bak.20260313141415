```python
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from . import face_capture  # replace with actual module name

def test_parse_args():
    args = face_capture.parse_args()
    assert isinstance(args, argparse.Namespace)

def test_parse_args_label():
    with patch('sys.argv', ['script.py', '--label', 'test_label']):
        args = face_capture.parse_args()
        assert args.label == 'test_label'

def test_parse_args_output_dir():
    with patch('sys.argv', ['script.py', '--output-dir', 'test_output_dir']):
        args = face_capture.parse_args()
        assert args.output_dir == Path('test_output_dir')

def test_parse_args_count():
    with patch('sys.argv', ['script.py', '--count', '10']):
        args = face_capture.parse_args()
        assert args.count == 10

def test_parse_args_camera_index():
    with patch('sys.argv', ['script.py', '--camera-index', '1']):
        args = face_capture.parse_args()
        assert args.camera_index == 1

def test_parse_args_delay():
    with patch('sys.argv', ['script.py', '--delay', '0.5']):
        args = face_capture.parse_args()
        assert args.delay == 0.5

def test_parse_args_use_dnn():
    with patch('sys.argv', ['script.py', '--use-dnn']):
        args = face_capture.parse_args()
        assert args.use_dnn

def test_parse_args_use_alignment():
    with patch('sys.argv', ['script.py', '--use-alignment']):
        args = face_capture.parse_args()
        assert args.use_alignment

def test_collect_faces():
    with patch('face_capture.build_face_detector') as mock_build_face_detector:
        with patch('face_capture.ensure_dir') as mock_ensure_dir:
            with patch('cv2.VideoCapture') as mock_cv2_VideoCapture:
                with patch('cv2.imshow') as mock_cv2_imshow:
                    with patch('cv2.waitKey') as mock_cv2_waitKey:
                        with patch('cv2.destroyAllWindows') as mock_cv2_destroyAllWindows:
                            with patch('cv2.imwrite') as mock_cv2_imwrite:
                                face_capture.collect_faces(
                                    'test_label',
                                    Path('test_output_dir'),
                                    10,
                                    0,
                                    0.5,
                                    use_dnn=True,
                                    use_alignment=True,
                                )
                                mock_build_face_detector.assert_called_once()
                                mock_ensure_dir.assert_called_once()
                                mock_cv2_VideoCapture.assert_called_once()
                                mock_cv2_imshow.assert_called()
                                mock_cv2_waitKey.assert_called()
                                mock_cv2_destroyAllWindows.assert_called_once()
                                mock_cv2_imwrite.assert_called()

def test_collect_faces_no_dnn():
    with patch('face_capture.build_face_detector') as mock_build_face_detector:
        with patch('face_capture.ensure_dir') as mock_ensure_dir:
            with patch('cv2.VideoCapture') as mock_cv2_VideoCapture:
                with patch('cv2.imshow') as mock_cv2_imshow:
                    with patch('cv2.waitKey') as mock_cv2_waitKey:
                        with patch('cv2.destroyAllWindows') as mock_cv2_destroyAllWindows:
                            with patch('cv2.imwrite') as mock_cv2_imwrite:
                                face_capture.collect_faces(
                                    'test_label',
                                    Path('test_output_dir'),
                                    10,
                                    0,
                                    0.5,
                                    use_dnn=False,
                                    use_alignment=True,
                                )
                                mock_build_face_detector.assert_called_once()
                                mock_ensure_dir.assert_called_once()
                                mock_cv2_VideoCapture.assert_called_once()
                                mock_cv2_imshow.assert_called()
                                mock_cv2_waitKey.assert_called()
                                mock_cv2_destroyAllWindows.assert_called_once()
                                mock_cv2_imwrite.assert_called()

def test_collect_faces_no_alignment():
    with patch('face_capture.build_face_detector') as mock_build_face_detector:
        with patch('face_capture.ensure_dir') as mock_ensure_dir:
            with patch('cv2.VideoCapture') as mock_cv2_VideoCapture:
                with patch('cv2.imshow') as mock_cv2_imshow:
                    with patch('cv2.waitKey') as mock_cv2_waitKey:
                        with patch('cv2.destroyAllWindows') as mock_cv2_destroyAllWindows:
                            with patch('cv2.imwrite') as mock_cv2_imwrite:
                                face_capture.collect_faces(
                                    'test_label',
                                    Path('test_output_dir'),
                                    10,
                                    0,
                                    0.5,
                                    use_dnn=True,
                                    use_alignment=False,
                                )
                                mock_build_face_detector.assert_called_once()
                                mock_ensure_dir.assert_called_once()
                                mock_cv2_VideoCapture.assert_called_once()
                                mock_cv2_imshow.assert_called()
                                mock_cv2_waitKey.assert_called()
                                mock_cv2_destroyAllWindows.assert_called_once()
                                mock_cv2_imwrite.assert_called()

def test_collect_faces_camera_error():
    with patch('cv2.VideoCapture') as mock_cv2_VideoCapture:
        mock_cv2_VideoCapture.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError):
            face_capture.collect_faces(
                'test_label',
                Path('test_output_dir'),
                10,
                0,
                0.5,
                use_dnn=True,
                use_alignment=True,
            )
```