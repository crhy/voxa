"""issue #72 v11b: launch the sandbox app with INFO logging to a file."""
import logging
import runpy
import sys

sys.path.insert(0, "/app/lib/voxa")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
runpy.run_path("/app/lib/voxa/voxa.py", run_name="__main__")
