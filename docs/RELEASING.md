# Releasing Voxa

## One version, written once

`voxa.APP_VERSION` in `voxa/__init__.py` is the only place the version is written.
Everything else follows it:

| Where | How it follows `APP_VERSION` |
| --- | --- |
| Python package (`pyproject.toml`) | declared `dynamic`, read from `voxa.__init__.APP_VERSION` |
| About dialog | imports `APP_VERSION` |
| AppStream (`io.github.crhy.voxa.metainfo.xml`) | the newest `<release>` must equal it (checked) |
| Release tag | must be `v<APP_VERSION>` (checked when a tag is pushed) |
| Flatpak bundle | named `Voxa-<version>-x86_64.flatpak` (derived) |

`python tools/release_check.py` validates all of this, needs nothing installed, and is
run by the test suite and by CI. It fails the build if anything disagrees:

```text
python tools/release_check.py                 # validate
python tools/release_check.py --tag v0.1.2    # also validate a tag
python tools/release_check.py --print artifact  # Voxa-0.1.2-x86_64.flatpak
```

## Cutting a release

1. Bump `APP_VERSION`.
2. Add a new `<release version="..." date="...">` entry at the top of the AppStream
   `<releases>` list (newest first).
3. Run the tests and `python tools/release_check.py`.
4. Merge, then tag the commit `v<version>` and push the tag. CI validates the tag against
   the version, builds the Flatpak, and attaches `Voxa-<version>-x86_64.flatpak` to the
   GitHub release.

## Where updates come from

The bundle records `https://crhy.github.io/voxa/flatpak-repo/` as its update repository:
Voxa's own GitHub Pages site, not another project's. That address is what a user who
installs the `.flatpak` file will follow for updates.

**Status: enable Pages once, then the next release tag publishes it.**

1. In the repository settings, open **Pages** and set **Source** to **GitHub Actions**
   (one-time; there is no branch to configure).
2. Push a release tag (`v<version>`, see above). The `pages` job in the workflow builds
   the update repository (`flatpak build-update-repo`), adds `voxa.flatpakrepo`, and deploys
   both to `https://crhy.github.io/voxa/`.
3. Users can then add the remote from the `.flatpakrepo` file, and `flatpak update` follows
   the bundle's recorded repository URL.

Until the first tagged release is published nothing answers at that address, so updates
from a bundle installed before then will fail with a "could not load summary" message.
After the first publish, test in this order: first installation from the bundle, an update
to a newer bundle, downgrade/rollback expectations, AppStream metadata, screenshots and icons.
The repository is currently **unsigned**; adding a GPG key is a follow-up (issue #7 section 22).
