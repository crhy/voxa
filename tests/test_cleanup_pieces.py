"""Clean-up works piece by piece so a small model never sees a whole page at once."""

from voxa.agent.cleanup import clean_in_pieces, split_pieces


def test_pieces_join_back_to_the_text_exactly():
    text = "First line here.  Second sentence!\n\nA new paragraph? Yes.\nlast line without a full stop"
    assert "".join(split_pieces(text, 20)) == text
    assert "".join(split_pieces("", 20)) == ""


def test_pieces_respect_the_size_and_line_breaks():
    text = ("One two three four five. " * 40).strip() + "\nShort line.\n"
    pieces = split_pieces(text, 100)
    assert all(len(piece) <= 100 for piece in pieces)
    assert pieces[-1] == "Short line.\n"


def test_each_piece_is_edited_and_spacing_is_kept():
    text = "i went too the store.\n\n  their was no bread.  "
    edited, done, total = clean_in_pieces(text, lambda piece: piece.capitalize().replace("too", "to"))
    assert (done, total) == (2, 2)
    assert edited == "I went to the store.\n\n  Their was no bread.  "


def test_a_runaway_or_failing_piece_is_left_alone():
    text = "Good sentence one. " * 30 + "\nBad piece here.\n" + "Another fine line.\n"

    def ask(piece):
        if piece.startswith("Bad"):
            return piece * 50  # the model ran away
        if piece.startswith("Another"):
            raise RuntimeError("model unavailable")
        return piece.replace("Good", "Great")

    edited, done, total = clean_in_pieces(text, ask, 200)
    assert "Bad piece here.\n" in edited and "Another fine line.\n" in edited
    assert "Great sentence one." in edited and "Good" not in edited
    assert done == total - 2


def test_nothing_to_edit_counts_nothing():
    assert clean_in_pieces("\n\n -- \n", lambda piece: "X") == ("\n\n -- \n", 0, 0)
