"""Issue #72, task v11 step 6: `--talk-over` paces VoxaTest to interrupt her.

The live run used to wait 1.5 s (settle) after her reply and never spoke over her, so a real
barge-in never happened and her stop latency was never measured. `--talk-over` speaks the next
question settings.talk_over_seconds into a long reply (resolve_barge_at: a value >= 1.0 is
ABSOLUTE SECONDS) so VoxaTest barges in 2-4 s in; only cases that do not already barge get the
override, and the settle drops to 1.0 s.
"""

from voxatest.cases import TestCase
from voxatest.config import HarnessSettings
from voxatest.hearing import resolve_barge_at
from voxatest.runner import _case_barges


def test_settle_defaults_to_one_second():
    # The task: next prompt <= 1.0 s after her reply ends (was 1.5).
    assert HarnessSettings().settle_seconds <= 1.0


def test_talk_over_defaults_to_two_to_four_seconds_in():
    s = HarnessSettings()
    assert s.talk_over is False
    # "2-4 s in" - the default lands inside that band on a >= 4 s reply.
    assert 2.0 <= s.talk_over_seconds <= 4.0
    assert s.talk_over_min_reply_seconds >= 4.0


def test_talk_over_override_barges_absolute_seconds_into_a_long_reply():
    # The mechanism --talk-over relies on: an absolute value is seconds, not a fraction.
    onset, position = resolve_barge_at(3.0, reply_seconds=9.0)
    assert onset == 3.0
    assert position == "seconds:3.0"


def test_case_barges_is_true_only_when_a_barge_position_is_named():
    assert _case_barges(TestCase(id="a", category="c", prompt="x", barge_at=0.5)) is True
    assert _case_barges(TestCase(id="a", category="c", prompt="x", barge_at="after_reply")) is True
    assert _case_barges(TestCase(id="a", category="c", prompt="x", barge_at=None)) is False
    assert _case_barges(TestCase(id="a", category="c", prompt="x", barge_at="")) is False
