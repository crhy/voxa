"""Tests must never reach the real sound server.

A test run that started the real window once loaded an echo-cancel device into the developer's PulseAudio and
switched the default audio output. Pointing PULSE_SERVER at nothing makes every pactl/parec call fail harmlessly.
"""

import os

os.environ["PULSE_SERVER"] = "unix:/nonexistent-voxa-tests"
os.environ["VOXA_NO_WELCOME"] = "1"
