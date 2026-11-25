"""Why attribution still reads silent mid-reply: the render map's anchors are AHEAD of the mic.

turn_reference_window(moment): end = min(_index_at_time(moment), last_index), end=max(0,..);
when _index_at_time(moment) is negative (the sink clock stamped audible_at later than the mic's
moment), end=0 and the window is FRONT-PADDED with silence -> reference_power ~ 0 ->
'reference active by definition while speaking: silent tap, abstain'.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from voxa.timeline import RenderTimeline

clock = [1000.0]
tl = RenderTimeline(sample_rate=16000, clock=lambda: clock[0])
tl.open("g1", "whole-reply")
tl.open_turn()
rate = 16000
# a 10 s reply: chunks arrive at the sink map 3 s in the future (big PipeWire queue)
for i in range(10):
    data = (0.3 * np.sin(2*np.pi*440*np.arange(rate)/rate)).astype(np.float32)
    tl.submit(data, generation="g1", path="whole-reply", now=clock[0] + 3.0)
print("total", tl.total, "turn live", tl.turn_is_live(clock[0]))
for offset in (0.0, 1.0, 5.0):
    w = tl.turn_reference_window(moment=clock[0] + offset, length=1600)
    print(f"moment +{offset}: power={float(np.mean(w**2)):.6f} nonzero={np.count_nonzero(w)}")
