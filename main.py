import json
import re
from collections import deque
from functools import lru_cache
from typing import List

import g4f
from dotenv import dotenv_values
from g4f import Provider
from IrcBot.bot import MAX_MESSAGE_LEN, Color, IrcBot, Message, utils

config = dotenv_values()
NICK = config["NICK"]
SERVER = config["SERVER"]
CHANNELS = json.loads(config["CHANNELS"])
PORT = int(config.get("PORT") or 6667)
PASSWORD = config["PASSWORD"] if "PASSWORD" in config else None
SSL = config["SSL"] == "True"

COMMANDS = [
    (
        "list",
        "List all providers",
        "Lists available providers with their info like model and url. Same as providers",
    ),
    (
        "providers",
        "List all providers",
        "Lists available providers with their info like model and url. Same as list",
    ),
    ("gpt3", "Generate text with GPT-3", "Generates text with GPT-3.5"),
    ("gpt4", "Generate text with GPT-4", "Generates text with GPT-4"),
    ("gpt", "Generate text with GPT-4", "Generates text with GPT-4. Same as gpt4"),
    (
        "clear",
        "Clears context",
        "Clears context for the user. Starts a fresh converstaion.",
    ),
]

model_map = {
    "gpt3": "gpt-3.5-turbo",
    "gpt4": "gpt-4",
    "gpt": "gpt-4",
}


utils.setPrefix("!")
utils.setHelpHeader(
    "GPT bot! Generate text using gtp4free. Context is saved for each user individually and between different providers."
)

# Load all modules from Provider.Providers
providers = [getattr(Provider, provider) for provider in dir(Provider) if not provider.startswith("__")]
providers = [
    provider
    for provider in providers
    if hasattr(provider, "model") and provider.model is not None and provider.needs_auth is False
]


def get_profider_name(provider):
    return provider.__name__.split(".")[-1]


command_to_provider = {get_profider_name(provider).lower(): provider for provider in providers}
print(command_to_provider)


for provider in providers:
    name = get_profider_name(provider)
    model = provider.model
    url = provider.url
    COMMANDS.append(
        (
            name.lower(),
            f"Generate text with {name}",
            f"Generates text with {name}. {model=} {url=}",
        )
    )


def ai_respond(messages: list[dict], model: str, provider=None) -> str:
    """Generate a response from the AI."""
    return g4f.ChatCompletion.create(model, messages, provider=provider, stream=False)


def preprocess(text: str) -> List[str]:
    """Preprocess the text to be sent to the bot.

    Consider irc line limit
    """
    return [text[i : i + MAX_MESSAGE_LEN] for i in range(0, len(text), MAX_MESSAGE_LEN)]


def generate_formatted_ai_response(nickname: str, text: str) -> List[str]:
    """Format the text to be sent to the channel."""
    lines = []
    for line in text.splitlines():
        if len(line) > MAX_MESSAGE_LEN:
            lines.extend(preprocess(line))
        else:
            lines.append(line)
    lines[0] = f"{nickname}: {lines[0]}"
    lines.append(f"{nickname}: --------- END ---------")
    return lines


def format_provider(provider: Provider) -> str:
    """Format the provider."""
    name = get_profider_name(provider)
    model = str(provider.model)[:64]
    url = provider.url
    available = Color("Yes", fg=Color.green).str if provider.working else Color("No", fg=Color.red).str
    return f"{name} {model=} {url=} -- available: {available}"


def list_providers(_, message: Message) -> list[str]:
    """List all providers."""
    return [message.nick + ": " + m for m in [format_provider(p) for p in providers]]


@lru_cache(maxsize=2048)
def get_user_context(nick: str) -> list[dict]:
    """Get the user context."""
    return deque([], maxlen=1024)


async def parse_command(bot: IrcBot, match: re.Match, message: Message, model: str = None):
    context = get_user_context(message.nick)
    text = message.text
    m = re.match(r"^!(\S+) (.*)$", text)
    if m is None or len(m.groups()) != 2:
        return f"{message.nick}: What?"

    if model is not None:
        provider = None
    else:
        command = m.group(1)
        provider = command_to_provider.get(command)
        if provider is None:
            provider = command_to_provider.get(command.lower())

        if provider is None:
            return f"{message.nick}: Provider '{command}' not found. Try !list or !providers."
        model = provider.model[0]

    text = m.group(2)
    context.append({"role": "user", "content": text})
    response = ai_respond(list(context), model, provider=provider)
    context.append({"role": "assistant", "content": response})
    print(f"{response=}")
    return generate_formatted_ai_response(message.nick, response)

async def clear_context(bot: IrcBot, match: re.Match, message: Message):
    get_user_context(message.nick).clear()
    return f"{message.nick}: Context cleared."

async def onConnect(bot: IrcBot):
    for channel in CHANNELS:
        await bot.join(channel)


if __name__ == "__main__":
    for command, help, command_help in COMMANDS:
        lower_name = command.lower()
        if command in ["list", "providers"]:
            func = list_providers
        elif command == "clear":
            func = lambda _, message: get_user_context(message.nick).clear()
        elif command in ["gpt3", "gpt4", "gpt"]:
            model = model_map[command]

            async def _func(bot, match, message):
                return await parse_command(bot, match, message, model=model)

            func = _func

        else:
            func = parse_command

        utils.arg_commands_with_message[lower_name] = {
            "function": func,
            "acccept_pms": True,
            "pass_data": False,
            "help": help,
            "command_help": command_help,
            "simplify": None,
        }

    bot = IrcBot(SERVER, nick=NICK, port=PORT, use_ssl=SSL, password=PASSWORD)
    bot.runWithCallback(onConnect)
