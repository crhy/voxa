# Voxa changelog

## 0.1.6 — in development

Everything below is on `main` after the 0.1.5 release. Items marked 🧪 pass their automated tests and still need
checking by ear and eye on a real desktop; `docs/CAPABILITIES.md` is the checklist with a phrase to try for each.

### Hearing and turn-taking

- **Voxa no longer interrupts herself.** A barge-in now needs two things: the microphone must be clearly louder
  than Voxa's own voice for six frames in a row (the level is learned while she speaks), and the words that were
  heard must not be her own sentence. The first 400 ms of every spoken piece ignore barge-in altogether, because
  the speaker onset is the loudest moment. (`voxa/bargein.py`)
- **A single "stop", "wait" or "Voxa" always counts as you interrupting**, unless she is saying that very word
  herself.
- **"Say …" is no longer typed into the focused window.** "Say hello to Bob" is spoken; only "type …", "write …",
  "dictate …" and "enter …" type.
- **Voxa's own echo canceller** (`voxa/anc.py`): estimates the delay between what the computer plays and what the
  microphone hears, then subtracts the played sound with an adaptive (NLMS) filter. It stays stable through silent
  pauses and never outputs louder than the microphone. A calibration routine (`voxa/anccal.py`) measures how many
  decibels of the computer's own sound are removed. (#72) 🧪
- Echo cancellation follows the microphone chosen in Voxa rather than the system default.
- A request may now be up to 20 seconds long.

### Sound devices

- **Voxa no longer adds a sound device or changes your audio output.** Through 0.1.5, the "system" echo
  mode loaded PulseAudio's echo-cancel module, which showed up in Sound Preferences as "Voxa-echo-cancelled-output"
  (mono, 32 kHz), became the default output and could be left behind after Voxa exited. That mode is removed; Voxa
  only listens to the output's monitor. On start, a device left behind by an older build is removed and the real
  output is restored. The automated tests are now also barred from the real sound server: test runs during
  development were the reason the output "kept switching". 🧪

### On the screen

- **The talking head follows you out of the window.** When Voxa is not the focused window, a round cut-out of the
  character (Medium lip movement) floats in the top-right corner of the monitor, above the program you are using,
  with a one-line status underneath. It never takes the keyboard focus; clicking it brings Voxa back. X11 only.
  (#86) 🧪 — checked in the real app on a private test display; not yet seen on the Compiz desktop.
- **Tips in the top-left corner** that change with what you are doing: dictating, paused, music playing and so on.
  (#85) 🧪
- The caption reads **Dictating** while dictation is on, instead of Listening. (#84) 🧪

### Apps, windows and the desktop

- "Minimize Pluma", "maximize Pluma", "restore Pluma", "show the desktop". (#74) 🧪
- "Lock the screen" — Voxa pauses herself first, so she does not react to the room while locked. (#82) 🧪
- "Volume up", "volume down", "mute", "unmute", "set the volume to 40 percent". (#80) 🧪
- Compiz by voice: "rotate cube left/right", "zoom in", "zoom in more", "zoom out", "reset zoom", "zoom left/right".
  Voxa reads your own Compiz key bindings (from the host when sandboxed) and presses those. (#81) 🧪

### Writing and files

- "Dictation" on its own starts dictation, as do "start dictation", "take dictation" and "dictate this". (#87) 🧪
- "Save file", "load file", "close file", "new document": sends the application's own shortcuts and asks for a file
  name. (#78) 🧪
- "Clean up the text": selects the text in the foreground window, has the model fix spelling, punctuation, grammar
  and clarity, pastes it back and says "Text edited for clarity." An edit that looks wrong (far longer or shorter,
  or chatty) is discarded and your text is left untouched. It works on **the selection** when something is
  selected ("clean this up", "fix this paragraph") and on the whole text otherwise, and it uses Voxa's own clipboard
  instead of the `xclip` tool, which is not installed by default; your clipboard contents are put back afterwards.
  (#77) 🧪
- **File management by voice**: "copy report from Downloads to Documents", "move holiday video from Downloads to
  Videos", "delete old notes from Documents" (goes to the trash, no confirmation — it can be taken back out),
  "empty trash", "find the file called budget", "find files bigger than 2 gigabytes", "open the Downloads folder",
  "what's in my Documents?", "rename report in Documents to final report", "create a folder called Taxes in
  Documents". File names are matched by how they sound, so "report final" finds `Report_Final.pdf`. Works on
  machines without an XDG user-dirs file, and the search skips hidden folders so it finishes in seconds. (#79) 🧪
- **Post a GitHub issue by voice**: "post an issue to GitHub for Spaced Linux" — Voxa asks for the title, then the
  description (say "stop dictation" to finish; "scratch that" removes the last sentence; "cancel" drops it), then
  opens GitHub's new-issue page already filled in for you to check and submit. The project is looked up among your
  own GitHub repositories by name or description; a stranger's project is never opened on a loose match. (#83) 🧪
- "Close file" now closes the document (Ctrl+W). It used to send Ctrl+Q, which quits the whole application.

### Music and video

- **"Play the latest video from my channel"** now works: say "my YouTube channel is …" once (spelling it out letter
  by letter is fine) and Voxa remembers it. Before, "my YouTube channel" was searched as if it were a channel's
  name. A channel that cannot be found is said out loud instead of failing silently. (#32) 🧪

### First start

- **A welcome on the very first start**: Voxa greets you by her character's name, out loud and in a small dialog,
  and offers the one thing that is missing — installing Ollama, or downloading the smallest model
  (`qwen2.5:0.5b`) — then tells you how to talk to her. After the install she goes straight on to the model
  download, and says "You're all set" when the first model is in. A tutorial link appears once the video exists.
  Still to come: the microphone and noise-cancellation check. (#76) 🧪

### Documentation

- `docs/CAPABILITIES.md`: a living list of everything Voxa can do, with a phrase to try and a tested / untested /
  not-working mark for each.
- This changelog.
- The website link in the app's store and About information points to voxaai.me. (#33)

### Under the hood

- The command bench (`python3 -m voxatest commands`) grew with every new phrase and must stay at 100%.
- Test suite: about 1,900 tests at the time of writing (0.1.5 shipped with about 1,650).
