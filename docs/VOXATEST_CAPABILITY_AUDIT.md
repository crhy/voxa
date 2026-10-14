# Voxa capability audit

This report measures Voxa's deterministic command router. It does not exercise the microphone, speech recognition, model backend, network, or desktop side effects. Use the spoken `voxatest run` suite for those end-to-end paths.

Result: 38400 checks, 38400 passed, 0 failed (pass rate 100.0%). 384 unique cases repeated 100 time(s).

## Throughput

- Elapsed: 3.111247 seconds
- Rate: 12,342.3 checks/second (44,432,347 checks/hour)
- Required minimum: 1,000 checks/hour
- Throughput result: **PASS**

## What works in the deterministic command router

| Category | Checks | Passed | Failed | Status |
| --- | ---: | ---: | ---: | --- |
| apps | 1300 | 1300 | 0 | PASS |
| browser | 2200 | 2200 | 0 | PASS |
| compiz | 600 | 600 | 0 | PASS |
| contacts | 400 | 400 | 0 | PASS |
| dictation | 1000 | 1000 | 0 | PASS |
| files | 4600 | 4600 | 0 | PASS |
| heard | 900 | 900 | 0 | PASS |
| help | 600 | 600 | 0 | PASS |
| home | 2000 | 2000 | 0 | PASS |
| images | 2000 | 2000 | 0 | PASS |
| keys | 2200 | 2200 | 0 | PASS |
| media | 5500 | 5500 | 0 | PASS |
| question | 2200 | 2200 | 0 | PASS |
| reading | 700 | 700 | 0 | PASS |
| reminders | 1500 | 1500 | 0 | PASS |
| settings | 900 | 900 | 0 | PASS |
| typing | 1500 | 1500 | 0 | PASS |
| web | 4200 | 4200 | 0 | PASS |
| windows | 2500 | 2500 | 0 | PASS |
| youtube | 1600 | 1600 | 0 | PASS |

## What does not work

No deterministic router failures were found.

## Wishlist (not handled yet)

None.

## Coverage limits

A passing row proves that text was routed to the expected registered tool with valid arguments. It does not prove that the target application accepted the action. Hardware, audio, model, network, and desktop integration results belong in the spoken reports and `docs/CAPABILITIES.md`.
