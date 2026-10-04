from voxa.agent.hearing import normalize, strip_wake


def test_transcripts_normalize():
    cases = [
        ("Voxa Closed Brutal Chess.", "close Brutal Chess"),
        ("Clothes Brutal Chess.", "close Brutal Chess"),
        ("those cards with cats.", "close cards with cats"),
        ("Lupin Brutal Chess.", "open Brutal Chess"),
        ("opened my email", "open my email"),
        ("openly breathes. Writer", "open LibreOffice Writer"),
        ("opened Libra Office right", "open LibreOffice Writer"),
        ("Open Libra Office, right?", "Open LibreOffice Writer"),
        ("Open the GAMP.", "Open GIMP"),
        ("Please some John Barry", "play some John Barry"),
        ("Voxet start dictation.", "start dictation"),
        ("Vox, save the file.", "save"),
        ("Boxa go offline", "go offline"),
        ("Vox A Stopdictation", "stop dictating"),
        ("Voxa, stop dictation.", "stop dictating"),
        ("Voxa.", ""),
        ("Open Space Bizarre ...", "Open Spaced Bazaar"),
        ("Close cloud", "Close Claude"),
    ]
    for heard, expected in cases:
        assert normalize(heard).lower().startswith(expected.lower()), (heard, normalize(heard))


def test_exact_results():
    assert normalize("Clothes Brutal Chess.") == "close Brutal Chess"
    assert normalize("Open the GAMP.") == "Open GIMP"
    assert normalize("Voxet start dictation.") == "start dictation"
    assert normalize("Vox A Stopdictation") == "stop dictating"
    assert normalize("Voxa.") == ""
    assert normalize("Close cloud") == "Close Claude"


def test_ordinary_sentences_unchanged():
    for sentence in [
        "those were the days",
        "please tell me a joke",
        "the cloud is grey",
        "I opened the door yesterday",
        "what is the capital of France",
        "play simon and garfunkel",
        "type hello world",
        "start dictating",
        "close this window",
        "how do I open gmail",
    ]:
        assert normalize(sentence) == sentence, normalize(sentence)


def test_strip_wake():
    cases = [
        ("hey voxa, open gmail", "open gmail"),
        ("Boxer", ""),
        ("open gmail", "open gmail"),
        ("... and then Voxa!", "... and then"),
        ("Voxa.", ""),
        ("Voxa Closed Brutal Chess.", "Closed Brutal Chess."),
        ("ok vox, close the app", "close the app"),
        ("Boxa go offline", "go offline"),
        ("Vox A Stopdictation", "Stopdictation"),
        ("Voxa, save the file.", "save the file."),
        ("open the gamp", "open the gamp"),
        ("vauxa open gmail", "open gmail"),
    ]
    for heard, expected in cases:
        assert strip_wake(heard) == expected, (heard, strip_wake(heard))
