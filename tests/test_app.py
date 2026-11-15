from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.chdir(ROOT)
    return AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()


def slider(app, label):
    return next(item for item in app.slider if item.label == label)


@pytest.mark.parametrize('method', ['Dynamic programming', 'Greedy', 'Graph cut (max-flow)'])
def test_history_survives_interaction(app, method):
    app.selectbox[0].select(method).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state['carve_result'][0].copy()
    slider(app, 'Show removal step').set_value(2).run()
    assert not app.exception
    assert len(app.success) == 1 and len(app.tabs) == 5
    assert slider(app, 'Show removal step').value == 2
    np.testing.assert_array_equal(app.session_state['carve_result'][0], result)


def test_exactly_one_seam(app):
    slider(app, 'Target width (px)').set_value(63)
    slider(app, 'Target height (px)').set_value(48)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.session_state['carve_result'][1]) == 1
    assert all(item.label != 'Show removal step' for item in app.slider)


def test_no_removals(app):
    slider(app, 'Target width (px)').set_value(64)
    slider(app, 'Target height (px)').set_value(48)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.session_state['carve_result'][1]) == 0


def test_new_target_invalidates_old_result(app):
    app.button[0].click().run()
    slider(app, 'Target width (px)').set_value(60).run()
    assert not app.exception
    assert not app.success
    assert 'carve_result' not in app.session_state
    app.button[0].click().run()
    assert app.session_state['carve_result'][0].shape[1] == 60


@pytest.mark.parametrize('shape', [(1, 8, 3), (8, 1, 3), (1, 1, 3)])
def test_single_pixel_source_dimensions(monkeypatch, shape):
    monkeypatch.chdir(ROOT)
    original_open = Image.open

    def open_example(path, *args, **kwargs):
        if str(path).replace('\\', '/').endswith('assets/example.png'):
            return Image.fromarray(np.full(shape, 100, dtype=np.uint8))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, 'open', open_example)
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.success) == 1
