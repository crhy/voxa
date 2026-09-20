from voxa import mail


def test_parse_email_command_extracts_recipient_and_topic() -> None:
    cases = [
        ("draft an email to mom about dinner plans", ("mom", "dinner plans")),
        ("please write me an email regarding the invoice", ("", "the invoice")),
        ("send email to John saying we'll be late", ("John", "we'll be late")),
        ("compose the email to Alex about the meeting!", ("Alex", "the meeting")),
        ("write an email about renewal", ("", "renewal")),
        ("email to my landlord", ("my landlord", "")),
        ("draft an email", ("", "")),
        ("email", ("", "")),
    ]
    for prompt, expected in cases:
        request = mail.parse_email_command(prompt)
        assert request is not None, prompt
        assert (request.to, request.topic) == expected, prompt


def test_parse_email_command_rejects_plain_questions_and_other_commands() -> None:
    for prompt in (
        "how do I send an email?",
        "what is an email address",
        "write an email to mom then open firefox",
        "tell me about the email",
        "",
    ):
        assert mail.parse_email_command(prompt) is None, prompt


def test_build_prompt_repeats_the_request() -> None:
    assert mail.build_prompt(mail.EmailRequest("mom", "dinner plans")) == "Draft an email to mom about dinner plans."
    assert mail.build_prompt(mail.EmailRequest("", "")) == "Draft a short, generic email."
    assert mail.build_prompt(mail.EmailRequest("Alex", "")) == "Draft an email to Alex."


def test_parse_draft_splits_subject_and_body() -> None:
    subject, body = mail.parse_draft(
        "Subject: Dinner on Friday\nBody:\nHi Mom,\n\nAre you free Friday evening?\n\nLove, Sam"
    )
    assert subject == "Dinner on Friday"
    assert body == "Hi Mom,\n\nAre you free Friday evening?\n\nLove, Sam"


def test_parse_draft_is_case_insensitive_and_survives_a_missing_body_label() -> None:
    subject, body = mail.parse_draft("subject: Reminder\nbody:\nPlease send the invoice.")
    assert subject == "Reminder" and body == "Please send the invoice."


def test_parse_draft_falls_back_to_the_whole_reply() -> None:
    subject, body = mail.parse_draft("Sure! Here is a note about the meeting:\nIt starts at 9.")
    assert subject == ""
    assert body == "Sure! Here is a note about the meeting:\nIt starts at 9."


def test_compose_builds_the_xdg_email_command(monkeypatch) -> None:
    launched = []
    monkeypatch.setattr(mail.subprocess, "Popen", lambda command, **kwargs: launched.append(command))
    mail.compose("mom@example.com", "Dinner on Friday", "Hi Mom!")
    assert launched[0] == ["xdg-email", "--subject", "Dinner on Friday", "--body", "Hi Mom!", "mom@example.com"]
    launched.clear()
    mail.compose("", "", "")
    assert launched[0] == ["xdg-email"]
    launched.clear()
    mail.compose("mom@example.com", "", "Only a body")
    assert launched[0] == ["xdg-email", "--body", "Only a body", "mom@example.com"]
