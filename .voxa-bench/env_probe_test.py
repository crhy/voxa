import os, subprocess
def test_env():
    print({k: os.environ.get(k) for k in ("HOME","XDG_RUNTIME_DIR","PULSE_SERVER","DISPLAY","USER")})
    r = subprocess.run(["pactl","load-module","module-null-sink"],capture_output=True,text=True)
    print(r.returncode, repr(r.stdout), repr(r.stderr))
