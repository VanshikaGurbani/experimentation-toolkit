from pathlib import Path

from streamlit.testing.v1 import AppTest

from expkit.power import sample_size_proportions

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def metric_values(at):
    return {m.label: m.value for m in at.metric}


def test_default_values_match_calculator():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    s = sample_size_proportions(0.04, 0.005)
    vals = metric_values(at)
    assert vals["Control users"] == f"{s.n_control:,}"
    assert vals["Total users"] == f"{s.total:,}"


def test_changing_inputs_changes_result():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.number_input[0].set_value(10.0).run()
    s = sample_size_proportions(0.10, 0.005)
    assert metric_values(at)["Control users"] == f"{s.n_control:,}"


def test_relative_mde_and_split():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio[0].set_value("Relative (%)").run()
    at.number_input[1].set_value(10.0).run()
    at.slider[0].set_value(85).run()
    s = sample_size_proportions(0.04, 0.10, relative=True, ratio=85 / 15)
    assert metric_values(at)["Total users"] == f"{s.total:,}"


def test_impossible_inputs_show_error_not_crash():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.number_input[0].set_value(98.0).run()
    at.number_input[1].set_value(5.0).run()
    assert len(at.error) == 1 and not at.exception
