export XDG_DATA_HOME=/home/rhy/.local/share
cd /home/rhy/strata-fleet/repos/voxa
nice -n 10 flatpak-builder --force-clean --ccache --repo=.build/repo .build/rc10 io.github.crhy.voxa.yml > .voxa-bench/build-rc10.log 2>&1
echo "BUILD_RC=$?" >> .voxa-bench/build-rc10.log
