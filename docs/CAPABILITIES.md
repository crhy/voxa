# What Voxa can do

A living list of everything you can ask Voxa, with a phrase to try for each. It doubles as the **test checklist**:
work down it, tick what works, and file an issue for anything that does not.

Say **“Voxa”** first, then the request (or speak straight away in the follow-up window after a reply).

**Status key**

| Mark | Meaning |
| --- | --- |
| ✅ | Works — checked on a real desktop |
| 🧪 | Built and passing its automated tests — needs testing by ear and eye |
| ⚠️ | Partly working — see the note |
| ❌ | Known not to work yet |

Last updated: 7 October 2026 (development build after 0.1.5).

---

## Talking to Voxa

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Voxa” … then a question | Caption goes **Ready → Listening → Thinking → Speaking → Ready** | ✅ |
| Ask anything while she is answering | She stops and listens (barge-in) | ✅ |
| “Voxa, pause” / press **PAUSE** | Caption **Paused**; everything is ignored until you say “Voxa” | ✅ |
| “Voxa” (while paused) | Back to **Ready/Listening** | ✅ |
| “Pause the music” | Pauses the player, not Voxa | ✅ |
| Press **OFFLINE** | Microphone off, nothing is heard | ✅ |
| “Can I get that in German?” | Repeats the last answer in German with a German voice (12 languages) | ✅ |
| “Back to English” | Returns to the character's own language | ✅ |
| “Volume up” … “again” … “do that again” | Repeats the last command (never a delete, a send or a lock) | 🧪 |
| A short reply | She starts speaking while the answer is still being written | ✅ |
| “What can you do?” / “What can you do with files?” | A short spoken list, with phrases to try | 🧪 |

## The face

| Try | What should happen | Status |
| --- | --- | --- |
| Click the round portrait (top right) | Grid of 22 characters; picking one switches face and voice | 🧪 |
| Facial Quality → **Low** | Still portrait | ✅ |
| Facial Quality → **Medium** | Lips move with the words; blinks | 🧪 |
| Facial Quality → **High** | Neural face with head movement — needs an NVIDIA card and the face server (developer machine only for now, see #73) | ⚠️ |
| Lip sync | Lips neither early nor late | ⚠️ reported a split second late; adjustable lead is in |

## Questions and the web

| Try saying | What should happen | Status |
| --- | --- | --- |
| “What is the current price of Bitcoin?” | Searches the web, answers from the results, shows **Source:** under the answer | 🧪 |
| “Tell me about Spaced Linux” | Searches for the named thing instead of guessing | 🧪 |
| “How many wins do they have?” (after a sports question) | Follow-up search using the previous topic | 🧪 |
| “What's 532 plus 789?” | Answers directly, no search | 🧪 |

## Apps and windows

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Open Brutal Chess” / “Open LibreOffice” | App opens (names are matched by sound) | ✅ |
| “Close Brutal Chess” | App closes | ✅ |
| “Close the browser” | Closes Voxa's own browser only — never yours | 🧪 |
| “Close Brave” | Closes the browser you named | 🧪 |
| “Minimize Pluma” / “Maximize Pluma” / “Restore Pluma” | Window state changes | 🧪 |
| “Show the desktop” | Everything minimizes | 🧪 |
| “Lock the screen” | Screen locks | 🧪 |
| “Rotate cube right” / “rotate cube left” | Compiz cube turns (uses your Compiz key bindings) | 🧪 |
| “Zoom in” / “zoom in more” / “zoom out” / “reset zoom” | Compiz desktop zoom | 🧪 |
| “Zoom left” / “zoom right” | Moves the zoomed view | 🧪 |

## Music and video

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Play Dirty Old Town by the Pogues” | Plays in VLC, no ads | ✅ |
| “Play the second movement of Beethoven's seventh” | Plays the video in VLC | ✅ |
| “Stop music” | Stops **and closes** the player | ⚠️ works when heard; mis-heard over loud music (see #72) |
| “Pause the music” / “resume” / “next” | Player control | 🧪 |
| “Volume up” / “volume down” / “mute” / “unmute” | Computer volume | 🧪 |
| “Set the volume to 40 percent” | Computer volume set exactly | 🧪 |
| “My YouTube channel is …” | Remembers your channel | 🧪 |
| “Play the latest video from my channel” | Plays your newest upload in VLC | 🧪 |

## Finding and buying

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Get me the cheapest ticket to Hawaii” | Expedia flight search from your current city, cheapest first | 🧪 |
| “Find me a cheap hotel in Tokyo for 4 nights” | Expedia hotels, low to high | 🧪 |
| “Rent a car in Denver next week” | Expedia car search by price | 🧪 |
| “Cheapest used Toyota Tacoma under 20000” | Cars.com by list price | 🧪 |
| “Best price on a 4070 Ti Super” | Google Shopping, price ascending | 🧪 |
| “Give me directions to the San Diego Zoo” | Google Maps directions | 🧪 |
| “Where is the nearest gas station?” | Google Maps, nearest first | 🧪 |
| “Show me pictures of …” | Image search | ✅ |

## Browsing

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Open my email” | Gmail opens | ✅ |
| “Open Expedia” / “Open spacedlinux.com” | Site opens, known or not | 🧪 |
| “Search Wikipedia for …”, “click History”, “go back” | Voxa drives her own browser | 🧪 |

## Writing and files

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Start dictation” … “stop dictation” | Types what you say into the focused window; caption reads **Dictating** | 🧪 |
| “New line”, “comma”, “period” while dictating | Punctuation | ✅ |
| “Save file” / “load file” / “close file” / “new document” | Sends the app's own shortcuts and asks for a file name | 🧪 |
| “Clean up the text” | Fixes spelling, grammar and clarity in the selected text, or in the whole text of the foreground window when nothing is selected | 🧪 |
| “Copy report from Downloads to Documents” | Copies the file, no confirmation | 🧪 |
| “Move holiday video from Downloads to Videos” | Moves the file, no confirmation | 🧪 |
| “Delete old notes from Documents” | Goes to the trash, no confirmation | 🧪 |
| “Empty trash” | Empties the trash | 🧪 |
| “Find the file called budget” | Locates the file by name | 🧪 |
| “Find files bigger than 2 gigabytes” | Lists the oversized files | 🧪 |
| “Open the Downloads folder” | Opens the folder | 🧪 |
| “What's in my Documents?” | Reads the first few names | 🧪 |
| “Rename report in Documents to final report” | Renames the file, keeping the old extension when the new name has none | 🧪 |
| “Create a folder called Taxes in Documents” | Makes the new folder | 🧪 |
| “Undo that” / “put it back” (after a file action) | Reverses the last copy, move, rename, delete or folder made | 🧪 |
| “Post an issue to GitHub for Voxa” | Asks for a title, then a description (say “stop dictation” to finish), then opens GitHub's new-issue page filled in for you to submit | 🧪 |
| “Send an email to my mom” | Asks for her address the first time (and remembers it), then the subject, then you dictate the message; opens the draft in your mail program for you to press Send | 🧪 |
| “Email Bob about the meeting” | Voxa writes a draft about that topic and opens it for you to review | 🧪 |
| “Mom's email is mom at example dot com” / “what is Mom's email?” | Remembers and reads back email addresses | 🧪 |

## Time and routines

| Try saying | What should happen | Status |
| --- | --- | --- |
| “Set a timer for 5 minutes” / “Remind me at 3 pm to call Bob” | Reminder fires and is spoken | 🧪 |
| “Cancel my reminders” | Reminders cleared | 🧪 |
| “Good morning” (a routine you defined) | Runs its steps | 🧪 |

## Hearing

| Situation | What should happen | Status |
| --- | --- | --- |
| Voxa speaking at normal volume | She does not hear and interrupt herself (system echo cancellation) | ⚠️ still happens at higher volume |
| Loud music playing, you say “Voxa, stop music” | Heard correctly | ❌ adaptive noise cancellation is not built yet (#72) |
| “Fox, stop music”, “Xa, pause” (mis-heard wake word) | Treated as the command you meant | 🧪 |
| Background words (“Music”, “As”, “Up”) | Ignored, not sent to the AI | 🧪 |

## Models and setup

| Try | What should happen | Status |
| --- | --- | --- |
| Start Voxa with Ollama selected | Starts Ollama itself; models list within seconds | 🧪 |
| Preferences → Local AI → pull a model | Moving progress bar; list refreshes when done | 🧪 |
| Backend → Strata | Shows Strata's model when Strata is running | 🧪 |
| Backend → llama.cpp | Lists model files even before the server starts | 🧪 |
| High selected with a large model | Warns when the two will not fit on the graphics card | 🧪 |

## On the screen

| Look for | What should happen | Status |
| --- | --- | --- |
| Top-left tips | Change with what you are doing (dictating, paused, music playing …) | 🧪 |
| Small face when Voxa is not in focus | A round talking head floats in the top-right corner of the screen, above other programs, with a one-line status; click it to bring Voxa back. X11 only | 🧪 |

---

## Not built yet

Tracked as issues: adaptive noise cancellation (#72), installing the High face from inside the app (#73), first-boot
welcome (#76), slimming the
download (#75), messages, email triage, calendar, phone calls, purchases and the other assistant plans (#41–#67).
