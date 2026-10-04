# Voice commands

A reference for what you can say to Voxa once it is listening. Every command
below is one Voxa can carry out on your desktop. Simple commands run
instantly and need no AI model; multi-step requests are planned by the local
model and take a few seconds.

## Web

| Say | Voxa does |
|---|---|
| "open gmail", "show my email", "check my inbox" | Opens Gmail in your browser |
| "open https://example.com" | Opens that exact web page |
| "search the web for the weather today" | Runs a web search for the phrase you said |
| "compose an email" | Opens a new Gmail compose window |
| "send it" | Sends the email you have just dictated |

## YouTube and media

| Say | Voxa does |
|---|---|
| "play lo-fi beats on YouTube" | Searches YouTube and plays the top result |
| "search youtube for cat videos" | Searches YouTube without playing anything |
| "play simon and garfunkel" | Plays the artist on YouTube |
| "pause the music", "turn it down a bit" | Sends the matching media key to your player |

## Apps and windows

| Say | Voxa does |
|---|---|
| "open spotify" | Launches the named application |
| "close spotify" | Closes the application by name |
| "close this window" | Closes the window that is currently focused |
| "switch to spotify" | Focuses the window of the named application |

## Typing and dictation

| Say | Voxa does |
|---|---|
| "type hello there" | Types the words you said into the focused field |
| "start dictating" | Enters dictation mode: everything you say is typed |
| "stop dictating" | Leaves dictation mode |
| "scratch that" | Deletes the last dictated sentence |
| "send it" | Sends what you typed (for example a composed email) |

Dictation mode is meant for longer text. After "start dictating", speak
normally and Voxa types as you go. Spoken punctuation works: say "period",
"comma", "question mark" or "new line" and the matching character is typed.
"scratch that" removes the most recent sentence if you misspoke. "send it"
finishes the message wherever you are typing.

## Keys

| Say | Voxa does |
|---|---|
| "press enter" | Presses the Enter key |
| "press escape" | Presses Escape |
| "copy that" | Copies the selection (Ctrl+C) |
| "paste it" | Pastes the clipboard (Ctrl+V) |

Any key you can name, Voxa can press.

## Multi-step requests

You can chain actions in one sentence and Voxa carries them out in order:

| Say | Voxa does |
|---|---|
| "open gmail and then compose a message" | Opens Gmail, then opens compose |
| "close brave and open libreoffice writer" | Closes Brave, then launches Writer |
| "find cat videos on youtube and then press escape" | Searches YouTube, then presses Escape |
| "switch to brave after that close the window" | Focuses Brave, then closes the window |

These are not instant: the local model plans the steps, so expect a few
seconds before the first action runs.

## Questions

| Say | Voxa does |
|---|---|
| "what is the capital of France" | Answers from the local model |
| "when should I use dictation" | Explains how Voxa works |

Questions are handled by the local model and answered out loud; they do not
touch your desktop.

## Instant versus planned

Single commands ("open gmail", "press enter") are matched directly and run
immediately, with no model involved. Multi-step requests ("open gmail and
then compose a message") are broken into steps by the local model, which
takes a few seconds.

## Where the data goes

Every action Voxa performs is written to an action log on your machine:

- `~/.local/state/voxa/actions.jsonl` — one line per action, what was said
  and what was done. It stays local.
- `voxatest commands` — the test suite's catalog of the commands above, used
  to verify that phrasings map to the right tools.
