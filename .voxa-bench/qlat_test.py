import time, array
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
pipe = Gst.parse_launch(
    "appsrc name=src is-live=false block=false format=time "
    "audioconvert! audioresample! audio/x-raw,rate=48000! autoaudiosink name=sink sync=true")
src = pipe.get_by_name("src")
src.set_property("stream-type", 0)
pipe.set_state(Gst.State.PLAYING)
def q():
    lat = Gst.Query.new_latency()
    ok = pipe.query(lat)
    vals = lat.parse_latency()
    return ok, vals
print("before feed:", q())
for i in range(40):
    data = array.array("h", [0]) * 4800
    buf = Gst.Buffer.new_wrapped(data.tobytes())
    buf.pts = i * 100_000_000
    buf.duration = 100_000_000
    src.emit("push-buffer", buf)
    time.sleep(0.01)
    if i % 10 == 9:
        print("during feed:", q(), flush=True)
time.sleep(0.2)
for _ in range(3):
    print("after feed:", q(), flush=True)
    time.sleep(0.4)
pipe.set_state(Gst.State.NULL)
