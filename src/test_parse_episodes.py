from parse_episodes import _is_music_mix


def test_is_music_mix():
    string1 = "This is not a music mix podcast"
    string2 = "This is a music mix podcast feat song (Original Mix)"
    string3 = "This is a music mix podcast feat song (Original Remix)"
    assert _is_music_mix(string1) == False
    assert _is_music_mix(string2) == True
    assert _is_music_mix(string3) == True

