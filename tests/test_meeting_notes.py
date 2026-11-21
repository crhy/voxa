"""Meeting and call notes (crhy/voxa#65): the record, the archive, the recall phrases.

All of it is text and the standard library: no microphone, no model, no network.
"""

from __future__ import annotations

from datetime import date

from voxa.agent.meeting_notes import (
    FILE_NAME,
    Record,
    action_items,
    append_record,
    decisions,
    parse,
    record,
    relative_day,
    relative_span,
    search,
    speakers,
    spoken_actions,
    spoken_decisions,
    spoken_recall,
)

TODAY = date(2026, 11, 16)

SAMPLE = (
    "Dana: We agreed to book the dentist on Friday.\n"
    "Me: OK, and we'll use the cheaper crown.\n"
    "Dana: I'll call the office back about the insurance.\n"
    "Me: I will send the invoice to Sam tomorrow.\n"
    "Dana: Let's pick up the kids on the way.\n"
)


def test_the_archive_has_its_own_file() -> None:
    assert FILE_NAME == "Voxa Meeting Notes.md"


def test_decisions_and_action_items_split_the_sample() -> None:
    assert decisions(SAMPLE) == [
        "We agreed to book the dentist on Friday.",
        "OK, and we'll use the cheaper crown.",
    ]
    assert action_items(SAMPLE) == [
        "I'll call the office back about the insurance.",
        "I will send the invoice to Sam tomorrow.",
        "Let's pick up the kids on the way.",
    ]


def test_a_sentence_with_a_task_is_an_action_item_not_a_decision() -> None:
    assert decisions("We'll send the invoice tomorrow.") == []
    assert action_items("We'll send the invoice tomorrow.") == ["We'll send the invoice tomorrow."]


def test_speakers_come_from_the_name_prefixes() -> None:
    assert speakers(SAMPLE) == ["Dana", "Me"]
    assert speakers("no prefixes here at all") == []


def test_record_is_one_dated_block() -> None:
    block = record("dentist call", date(2026, 11, 15), SAMPLE)
    assert block.startswith("## 2026-11-15 | Dentist call\n")
    assert "Decisions:\n- We agreed to book the dentist on Friday.\n" in block
    assert "Action items:\n- I'll call the office back about the insurance.\n" in block
    assert "Speakers:\n- Dana\n- Me\n" in block
    assert "Transcript:\nDana: We agreed to book the dentist on Friday.\n" in block


def test_the_archive_round_trips() -> None:
    content = append_record("", record("Dentist call", date(2026, 11, 15), SAMPLE))
    content = append_record(content, record("School run", date(2026, 11, 14), "Sam: Can you pick me up?\nMe: I'll schedule it for tomorrow."))
    items = parse(content)
    assert [item.title for item in items] == ["Dentist call", "School run"]
    assert items[0].day == date(2026, 11, 15)
    assert items[0].decisions == ("We agreed to book the dentist on Friday.", "OK, and we'll use the cheaper crown.")
    assert len(items[0].actions) == 3
    assert items[0].speakers == ("Dana", "Me")
    assert items[0].transcript == SAMPLE.strip()
    assert items[1].decisions == ()
    assert items[1].actions == ("I'll schedule it for tomorrow.",)


def test_search_by_keyword_and_day() -> None:
    content = append_record("", record("Dentist call", date(2026, 11, 15), SAMPLE))
    content = append_record(content, record("School run", date(2026, 11, 14), "Sam: Can you pick me up?\nMe: I'll schedule it for tomorrow."))
    assert [item.title for item in search(content, keyword="dentist")] == ["Dentist call"]
    assert [item.title for item in search(content, keyword="crown")] == ["Dentist call"]
    assert [item.title for item in search(content, day=date(2026, 11, 14))] == ["School run"]
    assert search(content, keyword="nothing like this") == []
    assert len(search(content)) == 2


def test_relative_day_reads_the_past_forms() -> None:
    assert relative_day("yesterday", TODAY) == date(2026, 11, 15)
    assert relative_day("the day before yesterday", TODAY) == date(2026, 11, 14)
    assert relative_day("two days ago", TODAY) == date(2026, 11, 14)
    assert relative_day("3 days ago", TODAY) == date(2026, 11, 13)
    assert relative_day("last week", TODAY) == date(2026, 11, 9)
    assert relative_day("today", TODAY) == date(2026, 11, 16)
    assert relative_day("last friday", TODAY) == date(2026, 11, 13)
    assert relative_day("next week", TODAY) is None
    assert relative_day("the dentist call", TODAY) is None


def test_relative_span_widens_a_week_phrase() -> None:
    assert relative_span("yesterday", TODAY) == (date(2026, 11, 15), date(2026, 11, 15))
    assert relative_span("last week", TODAY) == (date(2026, 11, 9), date(2026, 11, 15))
    assert relative_span("last monday", TODAY) == (date(2026, 11, 9), date(2026, 11, 9))


def test_the_spoken_forms() -> None:
    content = append_record("", record("Dentist call", date(2026, 11, 15), SAMPLE))
    items = parse(content)
    assert spoken_recall(items, TODAY) == (
        "You have 1 meeting note for that: Dentist call yesterday. "
        "Dentist call: 2 decisions: We agreed to book the dentist on Friday. "
        "OK, and we'll use the cheaper crown. "
        "3 action items: I'll call the office back about the insurance. "
        "I will send the invoice to Sam tomorrow. Let's pick up the kids on the way."
    )
    assert spoken_decisions(items, TODAY) == (
        "2 decisions: We agreed to book the dentist on Friday. OK, and we'll use the cheaper crown."
    )
    assert spoken_actions(items, TODAY) == (
        "3 action items: I'll call the office back about the insurance. "
        "I will send the invoice to Sam tomorrow. Let's pick up the kids on the way."
    )
    assert spoken_recall([], TODAY) == "I have no meeting notes for that."
    assert spoken_decisions([], TODAY) == "I heard no decisions in those notes."
    assert spoken_actions([], TODAY) == "I heard no action items in those notes."


def test_a_record_is_a_plain_named_tuple_of_parts() -> None:
    saved = parse(record("Call", TODAY, "Nothing here at all."))[0]
    assert isinstance(saved, Record)
    assert saved.decisions == () and saved.actions == () and saved.speakers == ()
    assert saved.transcript == "Nothing here at all."
