# Voxa changelog

## 0.1.6 — in development

Everything below is on `main` after the 0.1.5 release. Items marked 🧪 pass their automated tests and still need
checking by ear and eye on a real desktop; `docs/CAPABILITIES.md` is the checklist with a phrase to try for each.

### Changes from the first hands-on test round (8 October)

Found by using a local build on a real desktop, each traced to a cause before it was changed.

- **WebRTC echo cancellation is now the default echo setting** (it was opt-in). With music playing, Voxa's own
  canceller left so much sound that the end of a sentence was never detected. The mode's pipeline was rebuilt with
  the correct GStreamer calls and checked inside the Flatpak; a missing loopback falls back to the plain microphone.
- **A real voice detector finds pauses** instead of loudness alone, with the loudness rule as the fallback. On a
  recording of speech over music the loudness rule marked 36% of the non-speech time as voice; the detector, 0%.
- **Interrupting her works again, and she no longer interrupts herself.** The check now waits for a word or two
  before judging (it used to look after a third of a second, when there was nothing to read); the loudness gate
  tolerates the dips between syllables; a lone “stop” is no longer thrown away as echo; and her own voice misheard
  into different words (“BLC doesn't seem” heard as “Bmc, done, themes”) is recognised by its letters.
- **Wake word in one breath.** “Voxa open Pluma” said without a pause is matched when Whisper gets the first
  consonant wrong (“Box open Pluma”). On 59 recorded lines: 55 woke her before, 57 now, no false wake on 43 others.
- **“Close Brave” closes only the browser window Voxa opened.** It used to close every matching window and then
  kill the whole app, taking the user's own windows with it. No program is force-killed after being asked to
  close any more, so unsaved work is not thrown away.
- **Save by voice.** “Save file” on a new document asks “What should I save the file as?”, types the answer into
  the save window and saves; an existing name asks before replacing; “never mind” cancels. New documents go to
  Documents. She no longer says “Saved the file” after merely opening the save window.
- **“Clean up the text” works on long documents.** Text is edited in pieces of up to 500 characters; a piece whose
  edit looks wrong is left as it was. (Sent whole, a 7,000-character document made the small model return 110,000.)
- **Honest “Listening”.** The caption also covers the follow-up window, which has one length (the setting, default
  now 10 s) instead of two timers. After the wake word alone she waits 12 s.
- **Smaller fixes:** the dictation reminder is spoken only the first two times and dictation ends with “Stopping
  dictation”; a character change answers “Jack here!”; turning the cube is silent; “maximize yourself”,
  “minimize all apps”, “show desktop” and “restore everything”; folders open in the file manager; a web address
  said as words (“open air ships dot com”) opens; walking, biking and transit directions; “open the website” after a
  search; “open CNN” opens the site when no app matches, and a weak match no longer opens the wrong app; answers
  after a web search are taken from the results; VLC keeps a window smaller than Voxa's; the main window no longer
  changes size on ACTIVE; the keyboard goes to a new document so dictation types into it; a slow cube spin.
- **Start-up noise removed:** the GTK theme warning about `overflow`, and the “Unclosed client session” messages
  when the app closed during a voice download.
- **Per-stage timings** (hearing, routing, acting, first audio) are written to the action log, to find lag.
- 🧪 Not yet confirmed on a desktop: the voice detector in daily use, the slow cube spin (needs Compiz), new-document
  focus in LibreOffice, and save-by-voice in LibreOffice.

### Hearing and turn-taking

- **WebRTC echo cancellation as a new echo mode "webrtc"** (now the default; see above): one capture pipeline
  where the loopback monitor of the default output feeds a `webrtcechoprobe` and the microphone branch runs
  `webrtcdsp` inside the app; falls back to the plain capture pipeline if the monitor or pipeline fails, and to
  "off" when the elements are missing. (`voxa/audio.py`, `voxa/config.py`, `voxa/window.py`) (#88) 🧪

- **Voxa picks a Whisper model that fits this computer** at the first-run welcome and in Preferences: Turbo only when
  a graphics card is usable, otherwise Base (Tiny on small machines); a heavy manual choice still loads but warns.
  (`voxa/speechfit.py`) (#94)
- **Voxa watches how fast dictation really runs** and, when a Whisper model can't keep up on a machine with no
  usable graphics card, offers a lighter model with a "Switch" toast; the median real-time factor is remembered
  per model and saved at most once a minute. (`voxa/speechfit.py`, `voxa/conversation.py`, `voxa/window.py`) (#94) 🧪
- **Voxa no longer interrupts herself.** A barge-in now needs two things: the microphone must be clearly louder
  than Voxa's own voice for six frames in a row (the level is learned while she speaks), and the words that were
  heard must not be her own sentence. The first 400 ms of every spoken piece ignore barge-in altogether, because
  the speaker onset is the loudest moment. (`voxa/bargein.py`)
- **A single "stop", "wait" or "Voxa" always counts as you interrupting**, unless she is saying that very word
  herself.
- **Voxa throws away echoes of her own voice** (`voxa/ownspeech.py`): she knows the sentence she is saying, so a
  mic transcript that is (part of) that sentence is dropped before it reaches the wake check, a prompt or a barge-in
  candidate; a real sentence spoken over her voice keeps its own words. (#88) 🧪
- **"Say …" is no longer typed into the focused window.** "Say hello to Bob" is spoken; only "type …", "write …",
  "dictate …" and "enter …" type.
- **Voxa's own echo canceller** (`voxa/anc.py`): estimates the delay between what the computer plays and what the
  microphone hears, then subtracts the played sound with an adaptive (NLMS) filter. It stays stable through silent
  pauses and never outputs louder than the microphone. A calibration routine (`voxa/anccal.py`) measures how many
  decibels of the computer's own sound are removed. (#72) 🧪
- Echo cancellation follows the microphone chosen in Voxa rather than the system default.
- A request may now be up to 20 seconds long.

### Conversation

- **The caption says what she is doing** while a command runs — "Looking for budget…", "Copying report to
  Documents…", "Searching for Hawaii…" — in the main window and under the floating head. (#39) 🧪
- **"Again" / "do that again"** repeats the last command ("volume up" … "again" … "again"). It never repeats a
  delete, a send, a lock or typed text. (#69) 🧪
- While you are dictating into a document, nothing you say can start another command by accident.
- **"What can you do?"** is answered from a fixed, true list instead of the AI model improvising, with phrases to
  try; "what can you do with files?", "help with music" and so on go into one area. 🧪
- **Read-back**: "read the selection to me", "what's on my clipboard?", "summarize this" (three sentences, from the
  text you selected), "what's open?", "what window is this?". (#68) 🧪

### New commands (quick wins)

- Internet radio by name or genre: "play Groove Salad", "play some ska radio", "what stations do you have?". (#91) 🧪
- "Spin cube" keeps the desktop cube turning until you stop it, then returns to the starting desktop. (#90) 🧪
- "Take a picture" with the webcam; "show me the picture". (#92) 🧪
- "How do I say … in Japanese?" — translated and spoken in that language's voice. (#66) 🧪
- "Good morning" / "brief me": date, weather and today's reminders. (#57) 🧪
- The hints, tips, welcome and help text name your own wake word after you change it. (#93) 🧪
- A status indicator beside the backend selector: Online, Starting… or Offline. (#27) 🧪

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
- **"Undo that" / "put it back"** reverses the last file command: a move goes back, a copy is removed, a rename is
  reversed, a deleted file comes back out of the trash, a new (empty) folder is removed. With no file command
  behind it, "undo that" is the focused program's own undo. (#79) 🧪
- **Post a GitHub issue by voice**: "post an issue to GitHub for Spaced Linux" — Voxa asks for the title, then the
  description (say "stop dictation" to finish; "scratch that" removes the last sentence; "cancel" drops it), then
  opens GitHub's new-issue page already filled in for you to check and submit. The project is looked up among your
  own GitHub repositories by name or description; a stranger's project is never opened on a loose match. She says
  which project it will go to before asking for the title. Your account is read from the `gh` tool, or tell her:
  "my GitHub name is …". (#83) 🧪
- **A dictated email**: "send an email to my mom" — Voxa asks for the address the first time and remembers it,
  asks for the subject, then you dictate the message ("scratch that" removes a sentence, "stop dictation" finishes,
  "cancel" drops it). The draft opens in your mail program for you to press Send; she no longer writes the message
  herself or claims to have sent it. "Email Bob about the meeting" (with a topic) still gets an AI-written draft.
  "Mom's email is mom at example dot com", "what is Bob's email?" and "forget Bob's email" manage the addresses
  she knows. (#31) 🧪
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

- The command bench (`python3 -m voxatest commands`) grew from 296 to 379 phrasings and must stay at 100%.
- Test suite: about 2,060 test cases at the time of writing; 370 new test functions since 0.1.5 (1,150 to 1,520).
