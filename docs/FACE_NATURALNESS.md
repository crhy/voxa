# Face naturalness: beating the uncanny valley for Voxa's talking 3D assistant

Research brief for the owner's priority: Voxa's 3D characters must feel natural, not uncanny.
Every rule below is mapped to the controls Voxa actually has:

- **Visemes (15):** `viseme_aa, E, I, O, U, PP, FF, TH, DD, kk, CH, SS, nn, RR, sil`
- **ARKit-style units:** `eyeBlinkLeft/Right`, `eyeSquintLeft/Right`, `eyeWideLeft/Right`,
  `browInnerUp`, `browDownLeft/Right`, `browOuterUpLeft/Right`,
  `mouthSmileLeft/Right`, `mouthFrownLeft/Right`, `mouthPressLeft/Right`,
  `cheekSquintLeft/Right`, `jawOpen`, `mouthClose`,
  `eyeLookUp/Down/In/Out Left/Right`
- **Head:** whole-head rotation (yaw/pitch/roll); **eyes:** separate eye meshes (gaze direction).

Values without a fetched source are marked "(rule of thumb)".

## 1. What triggers the uncanny valley in a talking head

The uncanny valley hypothesis (Mori): an entity that looks almost human but is not quite
human elicits revulsion; the response is worst in the "almost human" band, and **movement
amplifies the emotional response** (Uncanny valley article). The anomaly/attribute theory on
the same page: the effect is strongest when *something that looks human moves mechanically*
(or vice versa). Reviewers of near-realistic CGI describe exactly the failures a talking
assistant must avoid: "a coldness in the eyes, a mechanical quality in the movements"
(Final Fantasy: The Spirits Within) and "plastic skin dotted with pores" (Tintin).

Concrete triggers to defend against, each with its Voxa countermeasure:

| Trigger | Symptom | Countermeasure (controls) |
|---|---|---|
| Dead eyes | no blinks, locked stare | blink driver on `eyeBlinkLeft/Right` |
| Frozen upper face | mouth moves, brows/cheeks still | couple brows + cheeks to speech and expression |
| Perfectly regular motion | metronomic blinks/nods | randomize timing and amplitude (section 7) |
| Lips popping between shapes | viseme snaps | coarticulation + easing (section 4) |
| No breathing | chest/face rigid, lids heavy | idle cycle on lids, brows, jaw (section 6) |
| Symmetric timing | left/right identical, mirrored | asymmetric variants (section 5, 7) |
| Staring | eyes fixed on camera for whole utterance | gaze saccades + aversion (section 2) |

Tinwell's game/animation research on that page studies how *cross-modal* factors (facial
expression plus speech) exaggerate the uncanny in real-time characters — i.e. desync between
lips, expression and voice is itself a trigger. Never let expression and lip-sync drift apart.

## 2. Eyes

**Blink rate and duration.** From the Blinking article: intervals between blinks are generally
2–10 s, averaging around 17 blinks/min in a laboratory setting; when the eyes are fixed on one
object (reading) the rate drops to about 4–5/min. Blink duration averages 100–150 ms (UCL
researcher), 100–400 ms per the Harvard biological-numbers database; closures over 1000 ms read
as microsleeps.

Voxa targets:
- **At rest (idle, listening):** mean interval 3.5–4.5 s (rule of thumb, inside the 2–10 s band),
  duration 110–150 ms on `eyeBlinkLeft/Right`.
- **While speaking:** slightly denser, mean interval ~2.5–3.5 s (rule of thumb) — the character
  is "active"; while *listening attentively* use the sparser 4–7 s end of the band.
- **Never** a blink that closes for under ~80 ms (reads as a twitch) or over ~400 ms (reads as
  a wink/microsleep).

**Blink asymmetry and mechanics.** The lids close via one muscle group and open via another
(orbicularis oculi closes, levator palpebrae opens — Blinking article), so real blinks are not
symmetric ramps. Implement: close in ~60–70% of the duration, open in the remainder — fast close,
slower open. Drive `eyeBlinkLeft` and `eyeBlinkRight` with independent durations and a 10–40 ms
offset (rule of thumb) so the eyes never move in perfect lockstep.

**Double blinks.** Real people sometimes blink twice in quick succession (a ~100–150 ms blink,
~150–250 ms open, another blink) — use occasionally (~1 in 8 blink events, rule of thumb) at
phrase ends; it reads as thoughtfulness.

**Blinks as punctuation.** Place a blink at gaze shifts and at phrase/sentence ends rather than
uniformly: a blink aligned to a clause boundary looks like the character is "thinking". Keep
blinks out of the middle of viseme sequences.

**Saccades and fixations (Saccade article).** Real gaze is not a smooth drift: eyes make quick
jumps between fixation points. Humans exploring a face make **2–3 fixations per second**; saccade
duration is 20–200 ms depending on amplitude (20–30 ms typical for small reading saccades);
latency to an unexpected stimulus is ~200 ms. Peak speeds reach 700°/s for 25° saccades (10° ≈
300°/s, 30° ≈ 500°/s). For Voxa:
- Move gaze by jumping the eye-mesh direction to a new target in 30–80 ms (small amplitudes),
  then hold a fixation 300–1200 ms (rule of thumb). Never tween gaze over hundreds of ms —
  smooth-pursuit gaze looks mechanical.
- Micro-movements during fixation: 1–3° jitter every 0.5–2 s (rule of thumb; microsaccades of
  2–120 arcminutes occur during real fixation per the Saccade article). Drive with tiny
  `eyeLookIn/Out/Up/Down` deltas or eye-mesh rotation.
- Gaze shifts larger than ~20° are accompanied by a head movement, and **the eyes saccade first,
  the head follows more slowly** (VOR, Saccade article). So: eye-mesh rotation leads, head
  rotation trails by 100–200 ms (rule of thumb).

**Where people look.** While **thinking/answering internally**: avert briefly — the Eye contact
article's Stirling study found children who avoid eye contact while composing answers do better,
and "a blank stare likely indicates a lack of understanding". So when Voxa is composing a reply,
glance up-left or to the side for 0.5–1.5 s, then return. While **talking to the user**: hold
gaze most of the time. While **listening**: look at the user, with occasional small fixations
around the face (2–3/s is too fast for a screen character; use 1 fixation per 1–3 s, rule of
thumb).

**How much eye contact feels right.** Direct/mutual gaze is what humans attend to (the article
notes infants' attention and recognition are facilitated by direct gaze), but prolonged unbroken
staring reads as aggressive (the article notes staring is impolite and induces agitation in
primates). Target: hold eye contact ~60–70% of conversational time (rule of thumb), break it
with blinks, glances, or a slight head-down rather than by never looking at all.

## 3. Head

- **Small continuous motion:** never park the head at a static pose. Idle: yaw ±1–3°, pitch ±1–2°,
  roll ±1° (rule of thumb), on a slow 4–8 s cycle with per-axis phase offsets.
- **Nods tied to emphasis:** a small nod (pitch, 2–5°) on stressed words / question marks; a
  one-shot confirm-nod (5–10°) at the end of an affirmation. A head-tilt (roll 2–4°) on
  questioning or friendly lines.
- **Head leads gaze shifts** for large turns: when the gaze target is far, rotate head and eyes
  together with the eyes arriving first (see section 2); for small shifts keep the head still.
- The FACS page (via Computer facial animation) lists 8 action units for rigid 3D head
  movements — head pose is a first-class channel of expression, not decoration.

## 4. Mouth

- **Coarticulation** (Coarticulation article): anticipatory coarticulation means a following
  sound's features are assumed during the preceding sound; carryover means effects persist after.
  The Computer facial animation article states viseme influence is greatest at the viseme center
  and degrades with distance (dominance functions, Cohen & Massaro). Implement for Voxa: blend
  each viseme with its neighbors — hold a viseme's peak only ~60–80 ms around its center
  (rule of thumb), and pre-shape the mouth 40–120 ms **before** the sound (anticipation): e.g.
  start rounding `O`/`U` during the preceding consonant.
- **Closures on p/b/m:** `PP` must fully close the lips (`mouthPressLeft/Right` high, `mouthClose`
  moderate, `jawOpen` low) for 60–120 ms (rule of thumb), then release with a small overshoot
  settle. The Viseme article groups /p, b, m/ into one viseme — one closure shape serves all
  three; likewise `DD` covers /t, d, n, l/ and `kk` covers /k, ɡ, ŋ/ (velar, mouth nearly neutral,
  jaw barely opens).
- **Jaw vs lips:** vowel openness is mostly `jawOpen` + `mouthClose`/`viseme_*` lip shapes; do not
  drive vowel height by jaw alone — a jaw-only mouth looks hinged and mechanical.
- **Never fully close between words:** rest shape is a slightly parted mouth (`sil` with
  `jawOpen` ≈ 0.05–0.15, rule of thumb), not a sealed line. A mouth that seals at every pause
  strobes and reads as a robot.
- **Ease in/out:** every viseme transition gets an ease curve (section 7); no linear viseme-to-viseme
  pops. `sil` between phrases is a soft hold, not a snap to a neutral face.

## 5. Upper face and expression

- **Brow raises on stress:** raise `browInnerUp` (plus a touch of `browOuterUpLeft/Right`) on
  stressed syllables and new/important words, 0.1–0.3 for 200–400 ms (rule of thumb), easing back.
  `browDownLeft/Right` for emphatic/determined lines. A talking face whose brows never move has a
  frozen upper face — a primary uncanny trigger (section 1).
- **Slight asymmetry:** offset the two sides by 10–30% amplitude and 30–80 ms timing (rule of
  thumb); real faces are not mirrored machines.
- **Smile coupling:** the Computer facial animation article's control-rig example is explicit: a
  "smile" control acts on the mouth curve **and the eyes squinting** simultaneously. For Voxa:
  `mouthSmileLeft/Right` > 0 must drive `cheekSquintLeft/Right` ≈ 0.4–0.6× smile and
  `eyeSquintLeft/Right` ≈ 0.2–0.4× smile. A smile with dead, wide-open eyes is the classic fake
  smile. Microexpression scale: genuine emotional flickers last under ~0.5 s (Microexpression
  article: microexpressions typically last less than half a second; ordinary emotions 0.5–4.0 s)
  — keep expression *changes* in that range, not 2 s crossfades.
- FACS (FACS article): expressions decompose into action units = muscle contractions/relaxations,
  plus action descriptors for unitary movements like a jaw thrust. Voxa's ARKit units are a
  FACS-like basis; compose expressions as weighted AU combinations, never as one monolithic
  "face morph".

## 6. Body and breath

- **Breathing rhythm:** a slow cycle (4–6 s, rule of thumb) that subtly lowers the eyelids
  (+0.03–0.08 `eyeBlinkLeft/Right` as a *partial* lid droop, never a full close), drops the brows
  a hair, and parts/jaws the resting mouth by ±0.02–0.05. The blink article notes fatigued eyes
  blink more and lids are muscle-driven — heavy, still lids read as a mask.
- **Posture shifts:** weight shifts / small torso or head-height changes every 5–15 s (rule of
  thumb) during long idle; pair with a gaze fixation change so the body and eyes agree.
- During listening, a slow single nod every 4–8 s (2–4°) signals attention without bobbing.

## 7. Timing

- **Easing curves:** use ease-out-in (or cubic) on all blendshape ramps; linear interpolation is
  the mechanical signature reviewers noticed ("mechanical quality in the movements").
- **Variability:** no two blinks identical — sample duration from 100–160 ms, interval from
  2–10 s (Blinking article band), amplitude 0.7–1.0, with left/right offsets. Same for nods,
  brow flicks, viseme blends (±15% amplitude, ±20% duration, rule of thumb).
- **Reaction latency:** when responding to user input, delay gaze/head reactions ~150–300 ms
  (rule of thumb, consistent with the ~200 ms saccade latency in the Saccade article). Instant
  reactions look like a system that already knew — a tell for "too perfect".
- **Frame budget:** Condon's micro-level work used 1/25 s granularity; at 30–60 fps all of the
  above durations (20 ms saccades up to 4 s fixations) are representable; do not quantize
  animation to 0.5 s "poses".

## 8. Rendering

- **Eye highlights (catchlights)** and a moist cornea read as life; the uncanny-valley page's
  Final Fantasy critique ("coldness in the eyes") is exactly a missing-highlight failure.
- **Soft shadowing** around eyes, nose wings, jawline; hard uniform shading reads as plastic
  (Tintin critique: "plastic skin dotted with pores").
- **Skin not plastic:** subtle subsurface scattering / tonal variation; pores only with
  human-scale proportions. The uncanny-valley design-principles section states photorealistic
  human texture should only be paired with human facial proportions; stylized proportions with
  photorealistic texture look eerie.
- **Stylisation is safer than near-realism:** the valley sits in the *almost-human* band, so a
  deliberately stylized character (non-human proportions, non-photoreal material) starts on the
  empathetic side of the curve and cannot fall into it; near-realism starts inside the danger
  zone and every missing micro-motion (see sections 1–7) keeps it there. Recommend Voxa stay
  clearly stylized: big readable eyes, simple materials, and put the effort into *motion*
  (blendshapes animate cheaper than photoreal skin animates believably).

## 9. Prioritised checklist for Voxa (biggest gain first)

1. **Blink driver** — intervals 2–10 s (mean ~4 s idle, ~3 s speaking), duration 100–150 ms,
   fast close / slower open, L/R offset 10–40 ms, occasional double blink.
   Controls: `eyeBlinkLeft/Right`. Biggest single uncanny fix (dead eyes).
2. **Coarticulated visemes** — dominance-style blending, 40–120 ms anticipation, `PP` full
   closure 60–120 ms, rest mouth never sealed. Controls: 15 visemes + `jawOpen`, `mouthClose`,
   `mouthPressLeft/Right`.
3. **Gaze saccades + head lead** — jump gaze in 30–80 ms, fixate 0.3–1.2 s, microsaccade jitter
   1–3°, eyes lead head by 100–200 ms on shifts. Controls: eye meshes / `eyeLook*`, head rotation.
4. **Upper-face coupling** — brow flicks on stress (0.1–0.3, 200–400 ms), smile drives
   `cheekSquint` (0.4–0.6×) and `eyeSquint` (0.2–0.4×). Controls: brows, cheeks, squints.
5. **Breathing/idle cycle** — 4–6 s lid/brow/jaw micro-motion; posture shift every 5–15 s.
6. **Asymmetric timing + easing** — cubic ease on every ramp, ±15–20% amplitude/duration
   randomization, 150–300 ms reaction latency.
7. **Rendering pass** — catchlights, soft contact shadows, stylized-not-photoreal skin.

## Sources (all fetched successfully)

- https://en.wikipedia.org/wiki/Uncanny_valley
- https://en.wikipedia.org/wiki/Blinking
- https://en.wikipedia.org/wiki/Saccade
- https://en.wikipedia.org/wiki/Eye_contact
- https://en.wikipedia.org/wiki/Viseme
- https://en.wikipedia.org/wiki/Coarticulation
- https://en.wikipedia.org/wiki/Microexpression
- https://en.wikipedia.org/wiki/Facial_Action_Coding_System
- https://en.wikipedia.org/wiki/Facial_animation (served as "Computer facial animation")
