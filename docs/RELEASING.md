# Releasing Voxa

## One version, written once

`voxa.APP_VERSION` in `voxa/__init__.py` is the only place the version is written.
Everything else follows it:

| Where | How it follows `APP_VERSION` |
| --- | --- |
| Python package (`pyproject.toml`) | declared `dynamic`, read from `voxa.APP_VERSION` |
| About dialog | imports `APP_VERSION` |
| AppStream (`io.github.crhy.voxa.metainfo.xml`) | the newest `<release>` must equal it (checked) |
| Release tag | must be `v<APP_VERSION>` (checked when a tag is pushed) |
| Flatpak bundle | named `Voxa-<version>-x86_64.flatpak` (derived) |

`python tools/release_check.py` validates all of this, needs nothing installed, and is
run by the test suite and by CI. It fails the build if anything disagrees:

```text
python tools/release_check.py                 # validate
python tools/release_check.py --tag v0.1.1    # also validate a tag
python tools/release_check.py --print artifact  # Voxa-0.1.1-x86_64.flatpak
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

**Status: not published yet.** GitHub Pages is not enabled for this repository, so nothing
answers at that URL today; installs from the bundle work, updates will not until the
repository is published there. Before relying on updates, publish the OSTree repository to
Pages and then test, in this order: first installation from the bundle, an update to a
newer bundle, downgrade/rollback expectations, AppStream metadata, screenshots and icons.
