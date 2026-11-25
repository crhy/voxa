import subprocess, sys
sys.path.insert(0, ".")
exec(open(".voxa-bench/characters_live.py").read().split("def main")[0].split('"""')[2])
_set_config("aoife", "en-IE-EmilyNeural", "en")
_restart_app()
run = subprocess.run([sys.executable, "tools/live_reference_check.py", "--replies", "1",
                      "--per-reply-timeout", "40", "--voice", "aoife",
                      "--text-file", ".voxa-bench/text_aoife.txt"], capture_output=True, text=True, timeout=240)
print("exit", run.returncode); print(run.stdout[-1500:]); print("STDERR:", run.stderr[-1500:])
