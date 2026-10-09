from voxa.agent import radio


def test_three_new_stations():
    built = radio.catalogue()
    assert len(built) == 15
    by_name = {s.name: s for s in built}
    assert by_name["Illinois Street Lounge"].stream == "https://ice1.somafm.com/illstreet-128-mp3"
    assert by_name["Bossa Beyond"].stream == "https://ice1.somafm.com/bossa-128-mp3"
    assert by_name["Christmas Lounge"].stream == "https://ice1.somafm.com/christmas-128-mp3"


def test_forgiving_names():
    built = radio.catalogue()
    assert radio.find("cow poke radio", built).name == "Cowpoke Radio"
    assert radio.find("cowpoke", built).name == "Cowpoke Radio"
    assert radio.find("bossa beyond radio", built).name == "Bossa Beyond"
    assert radio.find("illinois street", built).name == "Illinois Street Lounge"
    assert radio.find("illinois street lounge somafm", built).name == "Illinois Street Lounge"
    assert radio.find("bossa beyond somafm", built).name == "Bossa Beyond"
    assert radio.find("christmas lounge somafm", built).name == "Christmas Lounge"
    assert radio.find("cowboy junk", built) is None
