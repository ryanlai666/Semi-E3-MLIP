from semi_mlip.train import TrainConfig
from scripts.resume_pilots import pilot_finished


def test_legacy_completed_pilot_is_not_restarted(tmp_path):
    config=TrainConfig(epochs=200)
    legacy={'epoch':169}  # A completed legacy checkpoint has no early_stopped key.
    assert not pilot_finished(tmp_path,legacy,config)
    (tmp_path/'timing.json').write_text('{"epochs_completed":170,"planned_epochs":200}')
    assert pilot_finished(tmp_path,legacy,config)


def test_explicit_terminal_pilot_states(tmp_path):
    config=TrainConfig(epochs=60)
    assert pilot_finished(tmp_path,{'epoch':59},config)
    assert pilot_finished(tmp_path,{'epoch':30,'early_stopped':True},config)
    assert not pilot_finished(tmp_path,{'epoch':30,'early_stopped':False},config)
