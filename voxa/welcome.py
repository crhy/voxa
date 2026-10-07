"""First-boot welcome copy: greeting, the one missing piece, and what to say."""

SMALLEST_MODEL = "qwen2.5:0.5b"
TUTORIAL_URL = ""  # filled in when the tutorial video exists; empty hides the link

HOW_TO = (
    "Press ACTIVE, then say Voxa, followed by what you need. "
    "For example: Voxa, what's the weather today? "
    "Say Voxa, pause, when you want me to wait."
)


def greeting(name: str) -> str:
    name = name or "Voxa"
    return (
        f"Hello, I'm {name}. It's lovely to meet you. "
        "I can answer questions, open your programs, play music, "
        "browse the web and type what you say."
    )


def next_step(backend: str, server_reachable: bool, model_count: int) -> str:
    if backend != "ollama":
        return "ready"
    if not server_reachable:
        return "install"
    if model_count == 0:
        return "model"
    return "ready"


def step_text(step: str) -> tuple[str, str, str]:
    if step == "install":
        return (
            "One thing first",
            "I think with a small AI program called Ollama, which runs on this computer, "
            "so nothing you say leaves it. I can install it for you now.",
            "Install Ollama",
        )
    if step == "model":
        return (
            "One thing first",
            f"I need a model to think with. The smallest one, {SMALLEST_MODEL}, "
            "runs on any computer and downloads in a minute.",
            "Download a model",
        )
    return ("You're all set", HOW_TO, "Get started")


def spoken(step: str, name: str) -> str:
    return greeting(name) + " " + step_text(step)[1]
