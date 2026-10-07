from voxa.agent.mailflow import (
    ASK_BODY,
    ASK_SUBJECT,
    CANCELLED,
    MailFlow,
    contact_key,
    resolve,
    spoken_address,
    wants_dictated_email,
)


def test_spoken_address_table():
    assert spoken_address("alice at example dot com") == "alice@example.com"
    assert spoken_address("Bob dot Smith at gmail dot com") == "bob.smith@gmail.com"
    assert (
        spoken_address("j underscore doe at mail dot example dot org")
        == "j_doe@mail.example.org"
    )
    assert spoken_address("alice@example.com") == "alice@example.com"
    assert spoken_address("hello there") == ""
    assert spoken_address("at dot com") == ""


def test_contact_key():
    assert contact_key("My Mom.") == "mom"
    assert contact_key("my mom") == "mom"
    assert contact_key("Mom") == "mom"


def test_resolve():
    contacts = {"mom": "mom@example.com"}
    assert resolve("my mom", contacts) == "mom@example.com"
    assert resolve("Mom", contacts) == "mom@example.com"
    assert resolve("alice at example dot com", contacts) == "alice@example.com"
    assert resolve("bob", contacts) == ""


def test_full_flow_my_mom_no_contacts():
    flow = MailFlow("my mom", {})
    assert flow.step == "address"
    assert flow.first_question() == (
        "What is the email address for my mom? Say it, or say skip."
    )
    assert flow.caption == "Email address"

    said, ready = flow.feed("mom at example dot com")
    assert said == f"Got it. {ASK_SUBJECT}"
    assert not ready
    assert flow.to == "mom@example.com"
    assert flow.learned == ("mom", "mom@example.com")
    assert flow.step == "subject"
    assert flow.caption == "Email subject"

    said, ready = flow.feed("hello mom")
    assert said == ASK_BODY
    assert not ready
    assert flow.subject == "Hello mom"
    assert flow.step == "body"
    assert flow.caption == "Email body"

    said, ready = flow.feed("hi mom")
    assert said == ""
    assert not ready
    said, ready = flow.feed("how are you")
    assert said == ""
    assert not ready

    said, ready = flow.feed("oops", control="undo")
    assert said == "Removed."
    assert not ready
    assert flow.parts == ["hi mom"]

    said, ready = flow.feed("happy birthday")
    assert said == ""
    assert not ready
    assert flow.parts == ["hi mom", "happy birthday"]

    said, ready = flow.feed("stop dictation", control="stop")
    assert ready is True
    assert said == "Your email to my mom is ready. Check it and press Send."
    assert "my mom" in said
    assert "press Send" in said
    assert flow.step == "done"
    assert flow.body == "Hi mom. Happy birthday."
    assert flow.caption == ""
    assert not flow.active


def test_known_contact_skips_address():
    flow = MailFlow("my mom", {"mom": "mom@example.com"})
    assert flow.step == "subject"
    assert flow.to == "mom@example.com"
    assert flow.first_question() == ASK_SUBJECT


def test_skip_at_address_leaves_to_empty():
    flow = MailFlow("my mom", {})
    said, ready = flow.feed("skip")
    assert said == ASK_SUBJECT
    assert not ready
    assert flow.to == ""
    assert flow.step == "subject"


def test_cancel_at_every_step():
    for to_spoken, contacts, seed in [
        ("my mom", {}, []),
        ("my mom", {"mom": "m@e.com"}, [("feed", "hi", None)]),
    ]:
        flow = MailFlow(to_spoken, contacts)
        for _action, text, control in seed:
            flow.feed(text, control)
        said, ready = flow.feed("cancel")
        assert said == CANCELLED
        assert not ready
        assert flow.step == "done"


def test_stop_at_subject_cancels():
    flow = MailFlow("my mom", {"mom": "mom@example.com"})
    said, ready = flow.feed("stop dictation", control="stop")
    assert said == CANCELLED
    assert not ready
    assert flow.step == "done"


def test_stop_with_empty_body_cancels():
    flow = MailFlow("my mom", {"mom": "mom@example.com"})
    flow.feed("hi")
    said, ready = flow.feed("stop", control="stop")
    assert said == CANCELLED
    assert not ready
    assert flow.step == "done"


def test_feed_after_done():
    flow = MailFlow("my mom", {"mom": "mom@example.com"})
    flow.feed("hi")
    flow.feed("hello there")
    said, ready = flow.feed("stop", control="stop")
    assert ready is True
    said, ready = flow.feed("anything")
    assert said == ""
    assert not ready


def test_caption_per_step():
    flow = MailFlow("my mom", {})
    assert flow.caption == "Email address"
    flow.feed("a at b dot com")
    assert flow.caption == "Email subject"
    flow.feed("hi")
    assert flow.caption == "Email body"
    flow.feed("stop", control="stop")
    assert flow.caption == ""


def test_wants_dictated_email():
    assert wants_dictated_email("mom", "") is True
    assert wants_dictated_email("bob", "the meeting") is False
