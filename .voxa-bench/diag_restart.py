import sys
sys.path.insert(0, ".")
src = open(".voxa-bench/characters_live.py").read().split('def main')[0].split('"""')[2]
exec(src)
_set_config("aoife", "en-IE-EmilyNeural", "en")
_restart_app()
r = _flatpak("run", "--user", "--command=gapplication", "io.github.crhy.voxa",
             "action", "io.github.crhy.voxa", "read-file", f"'{REPO}/.voxa-bench/text_aoife.txt'")
print("rc", r.returncode); print("OUT", r.stdout[:300]); print("ERR", r.stderr[:500])
