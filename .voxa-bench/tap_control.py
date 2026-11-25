"""v11b control: does audio actually flow through autoaudiosink in the sandbox, and
what child does it create on the state change? No tap involved."""
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)
p = Gst.parse_launch("audiotestsrc wave=sine freq=440 ! autoaudiosink name=sink")
r = p.set_state(Gst.State.PLAYING)
print("set_state:", r)
loop = GLib.MainLoop()
GLib.timeout_add_seconds(3, loop.quit)
loop.run()
res, state, pending = p.get_state(3 * Gst.SECOND)
print("get_state:", res, state, pending)
sink = p.get_by_name("sink")
print("sink:", sink, sink.get_factory().get_name() if sink else None)
try:
    print("children:", [c.get_factory().get_name() for c in sink.get_children()])
except Exception as exc:
    print("children err:", exc)
# what does the registry pick as the sink at all?
for name in ("pipewiresink", "pulsesink", "alsasink", "ximagesink"):
    print(name, Gst.ElementFactory.find(name) is not None)
print("GST_AUTOAUDIO... default order:")
reg = Gst.Registry.get()
for f in reg.get_feature_list():
    if f.get_name() in ("pipewiresink", "pulsesink", "autoaudiosink", "fakesink", "alsasink"):
        print(" ", f.get_name(), "rank", f.get_rank())
