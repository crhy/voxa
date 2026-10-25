from voxa.agent.intents import route


def test_cap86_gpu_status():
    assert route("what is my gpu temp").tool == "gpu_status"
    assert route("gpu temperature").tool == "gpu_status"
    assert route("how hot is my graphics card").tool == "gpu_status"


def test_cap86_cpu_status():
    assert route("how hot is my cpu").tool == "cpu_status"
    assert route("cpu temperature").tool == "cpu_status"
    assert route("what's the cpu temperature").tool == "cpu_status"


def test_cap86_system_health():
    assert route("is my computer healthy").tool == "system_health"
    assert route("system health").tool == "system_health"


def test_cap86_battery_level():
    assert route("what's my battery level").tool == "battery_level"
    assert route("battery level").tool == "battery_level"


def test_cap86_disk_space():
    assert route("how much disk space is left").tool == "disk_space"
    assert route("how much storage space is free").tool == "disk_space"
    assert route("how much disk space do i have").tool == "disk_space"
    assert route("how much storage space is left").tool == "disk_space"


def test_cap86_memory_free():
    assert route("how much memory is left").tool == "memory_free"
    assert route("how much ram do i have free").tool == "memory_free"


def test_cap88_open_site_gmail():
    assert route("can you open gmail for me").tool == "open_site"
    assert route("go to gmail").tool == "open_site"


def test_cap88_open_site_github():
    assert route("please open github").tool == "open_site"
    assert route("open github for me").tool == "open_site"


def test_cap88_image_search_northern_lights():
    assert route("show me pictures of the northern lights").tool == "image_search"
    assert route("find images of the northern lights").tool == "image_search"


def test_cap88_open_site_youtube():
    assert route("can you open youtube").tool == "open_site"
    assert route("go to youtube").tool == "open_site"


def test_cap88_open_site_wikipedia():
    assert route("please open wikipedia").tool == "open_site"
    assert route("open wikipedia for me").tool == "open_site"


def test_cap88_image_search_golden_retrievers():
    assert route("search for images of golden retrievers").tool == "image_search"
    assert route("find pictures of golden retrievers").tool == "image_search"


def test_cap88_open_url():
    assert route("open wikipedia dot org").tool == "open_url"
    assert route("navigate to wikipedia.org").tool == "open_url"


def test_cap91_daily_briefing():
    assert route("morning briefing").tool == "daily_briefing"
    assert route("give me my morning briefing").tool == "daily_briefing"


def test_cap93_where_am_i():
    assert route("what is my location").tool == "where_am_i"
    assert route("what's my location").tool == "where_am_i"


def test_cap93_local_weather():
    assert route("what's the weather here").tool == "local_weather"
    assert route("check the local weather").tool == "local_weather"


def test_cap93_local_weather_tokyo():
    assert route("what is the weather in Tokyo").tool == "local_weather"
    assert route("check weather in Tokyo").tool == "local_weather"
