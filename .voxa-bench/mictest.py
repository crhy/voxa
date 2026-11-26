import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
Gst.init(None)
p = Gst.parse_launch("pulsesrc ! audioconvert ! fakesink name=f")
count = [0]

def probe(pad, info):
    count[0] += 1
    return Gst.PadProbeReturn.OK

fp = p.get_by_name("f").get_static_pad("sink")
fp.add_probe(Gst.PadProbeType.BUFFER, probe)
loop = GLib.MainLoop()
bus = p.get_bus()
bus.add_watch(None) if False else None

def tick():
    print("t buffers=", count[0], "state=", p.get_state(0).state, flush=True)
    return True

GLib.timeout_add(1000, tick)
GLib.timeout_add(6000, loop.quit)
print("state rc:", p.set_state(Gst.State.PLAYING), flush=True)
loop.run()
print("buffers:", count[0])
msg = bus.poll(0, 0)
if msg:
    print("bus error:", msg.parse_error())
