# Voxa as a hands-free personal assistant (roadmap #68)

Tracking issue: [crhy/voxa#68](https://github.com/crhy/voxa/issues/68) — its body is the
roadmap; this file mirrors that build order and names the shipped components. #68 stays
the source of truth for the checklist. A 7 October 2026 owner comment on #68 reorders the
next steps after the hearing work (#88/#72, owned separately): quick wins first, then the
foundations, then the features that stand on them.

Each line is `- [ ] #NN <title> — what it does`; shipped ones say `built:` and name the
component. Items marked *0.1.9 chain* shipped on the `strata/f160-voxa-release-0-1-9-integration`
release chain and are not yet on `main`.

## Top priority
- [ ] #69 The fastest, most natural voice loop (measure, smarter end of speech, tiered models, streaming speech, natural turn-taking) — the top priority per #69: per-stage timings (hearing, routing, acting, first audio) in the action log and the "again" repeat are built; the end-of-speech, tiered-model and streaming-speech work continues on the hearing plan (#88/#72), not in this doc.
- [ ] #40 Real browser control (in progress) — see Foundations; it is the foundation the later phases stand on.

## Already built (on `main` or in progress)
- Voice-command tool layer, router, planner, dictation, action log, VoxaTest command bench (docs/VOICE_COMMANDS.md, `voxa/agent/registry.py`, `~/.local/state/voxa/actions.jsonl`)
- Program-name recognition (#37), Strata backend (#38), live activity display (#39)
- Since the issue body was written, many items below shipped: the 2026-10-07 comment names browser control, file management, email by dictation, issues by voice, system update, app install, read-back tools, notes, calculator and button pressing in any program (docs/CAPABILITIES.md "Pressing buttons in other programs").

## Foundations
- [ ] #40 Real browser control — built: Voxa drives her own browser ("Search Wikipedia for …", "click History", "go back" — CAPABILITIES.md "Browsing") and runs the "Finding and buying" searches; downloads land in `~/Downloads/Voxa` through the `download_file` tool.
- [ ] #41 Secrets store and personal profile — built (0.1.9 chain): `voxa/agent/secrets.py` keeps secrets in the system keyring (values never in settings, the action log or a spoken reply) and `Settings` holds the non-secret profile ("my address is …", "what address do you have?"), like the remembered-email-address and channel lines in CAPABILITIES.md.
- [ ] #42 Phone call engine — not built: CAPABILITIES.md "Not built yet" still lists phone calls; every Phase 4 item waits on it.
- [ ] #43 Decision: safeguards for actions that spend money — built (0.1.9 chain): one `spend_guard` in the tool layer — read-back, spoken confirm within 20 s, per-order and per-day caps in Preferences → Advanced — on top of the permission policy and confirmation rules in ARCHITECTURE.md; paying and buying stay refused until the guard confirms.

## Phase 1: needs no browser control
- [ ] #44 Home automation — built: `voxa/agent/homeassistant.py` + the home tools drive the hub from the url/token in ConfigStore (lights, switches, thermostat, scenes); covers, garage doors, locks and colour set-to/toggle added on the 0.1.9 chain.
- [ ] #55 Calendar and reminders — built: reminders and the calendar tools behind CAPABILITIES.md "Time and routines"; "move my dentist to Thursday at 4" and free-time search built on the 0.1.9 chain; CalDAV sync still open.
- [ ] #67 Routines and scheduled tasks — built: a spoken routine runs its steps ("Good morning"); putting routines on a weekday/time schedule in `routines.json` built on the 0.1.9 chain.
- [ ] #64 Find and organise anything on my computer — built: the `documents.py` file commands (find, open, rename, folders, trash, undo — CAPABILITIES.md "Writing and files"); searching inside file contents ("find files containing invoice") built on the 0.1.9 chain.
- [ ] #45 Bank statements (phase 1: from a provided PDF or CSV) — built (0.1.9 chain): the statement importer parses the user's own CSV/PDF text into the spending log with merchant category rules, month-over-month comparison and a written report.
- [ ] #62 Receipts, expenses and budget — built (0.1.9 chain): a receipt photo or scan is read with the desktop's text reader and filed in the same spending ledger the spoken expense commands use; the ledger itself is on main.
- [ ] #54 Make and send invoices — built (0.1.9 chain): one spoken sentence becomes an invoice document that is opened, through `documents.py` write-and-open; sending still rides the dictated-email flow.
- [ ] #65 Meeting and call notes — built (0.1.9 chain): a recording becomes a dated meeting note in Documents (transcript, decisions, action items) and "what did we decide yesterday?" searches the archive.
- [ ] #66 Live translation and interpreter mode — partly built: `say_in_language` on main ("How do I say … in Japanese?", 12 languages); the two-sided interpreter session (start/stop/switch, turn-by-turn) built on the 0.1.9 chain.

## Phase 2: on top of browser control and the profile
- [ ] #56 Email: read, summarise, reply, clean up — partly built: read-only IMAP (unread counts, senders, subjects) on the 0.1.9 chain, and the draft flow on main ("Email Bob about the meeting", CAPABILITIES.md); summarise, reply and triage still open ("Not built yet: email triage").
- [ ] #57 Daily briefing — built: "Good morning"/"brief me" speaks date, weather and today's reminders (CAPABILITIES.md); the bills and budget line added on the 0.1.9 chain; important email and deliveries still open.
- [ ] #47 Book a restaurant table (online) — built (0.1.9 chain): the reservation tools open OpenTable/Resy/Tock with party, date and time pre-filled and say which fields they could not fill; nothing submits — the #43 guard owns that step.
- [ ] #52 Track existing orders and deliveries — built (0.1.9 chain): tracking numbers out of pasted text into a spoken-back orders ledger ("where is my filament order?", "what's arriving this week?").
- [ ] #63 Fill in forms and documents — partly built (0.1.9 chain): `form_read`/`form_fill`/`form_submit` type the profile's name, address and email into an open page's matching fields and submit only behind a yes-confirmation; the PDF/desktop-form half stays open.
- [ ] #58 Send and read messages — not built: CAPABILITIES.md "Not built yet" lists messages.
- [ ] #45 Bank statements (phase 2: fetch from the bank) — not built: waits on browser control plus the #41 profile and the owner's bank question below.
- [ ] #60 Bills and subscriptions — built (0.1.9 chain): recurring charges detected in the spending log, next due dates rolled forward, and a bill reminder per charge ("what bills are due this week?", "what subscriptions do I pay for?").

## Phase 3: spending money
- [ ] #46 Buy Bitcoin through an exchange API — partly built (0.1.9 chain): price, balance and recent orders read from the user's own exchange (Kraken, Coinbase, Bitstamp); the buy phrase is recognised and refused out loud until the #43 guard confirms a trade.
- [ ] #51 Buy things online — partly built: the product-search read-back (top three, cheapest first) on the 0.1.9 chain; cart and checkout still open behind #43.
- [ ] #61 Groceries and food delivery — partly built: the shopping list is on main; the order read-back (items, total band, services) is on the 0.1.9 chain; ordering still open.
- [ ] #50 Order a ride — partly built (0.1.9 chain): the ride-quote tool opens the provider with pick-up and destination pre-filled and speaks the estimate; requesting and scheduling stay behind #43.
- [ ] #49 Find and book hotel rooms — partly built (0.1.9 chain): the hotel tool reads back three booking options with price bands; reserving still open.
- [ ] #48 Search and buy airline tickets — search built: "Get me the cheapest ticket to Hawaii" runs an Expedia search cheapest-first (CAPABILITIES.md "Finding and buying"); buying still open behind #43.

## Phase 4: on top of the phone call engine
- [ ] #47 Book a restaurant table (by phone) — not built: waits on #42.
- [ ] #59 Make appointments by phone — not built: waits on #42.
- [ ] #53 Book a musician or band — not built: waits on #42.

## Open questions for the owner
1. Which bank, and does it offer an export or an aggregator connection? (#45)
2. Which exchange? (#46)
3. Which smart-home devices and hub? (#44)
4. Purchases: read-back confirmation, a daily cap, both, or neither? (#43)
5. Phone calls: a VoIP/SIP account, or through the mobile? (#42)
