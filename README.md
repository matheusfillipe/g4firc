## GPT4free IRC bot

By using this repository or any code related to it, you agree to the [legal notice](./LEGAL_NOTICE.md). The author is not responsible for any copies, forks, or reuploads made by other users. This is the author's only account and repository. To prevent impersonation or irresponsible actions, you may comply with the GNU GPL license this Repository uses.

This (quite censored) New Version of gpt4free, was just released, it may contain bugs, open an issue or contribute a PR when encountering one, some features were disabled.
Docker is for now not available but I would be happy if someone contributes a PR. The g4f GUI will be uploaded soon enough.

## Setup

* https://github.com/xtekky/gpt4free
* `python -m venv .venv`
* `source .venv/bin/activate`
* `pip install -r requirements.txt`
* Whatever else is needed to make g4f works (currently needs to be locally build, pip pakcage might not work)
* `cp .env.example .env`
* run `main.py` (directly or with pm2)

## Workaround for broken g4f package

As of the time of this commit g4f pip package installation is broken and some providers are missing. You can ignore the step above to pip install and do it manually or fix it with something like:

```bash
git clone https://github.com/xtekky/gpt4free.git
cd gpt4free
cp -r g4f/Provider/Providers/* ../.venv/lib/python3.10/site-packages/g4f/Provider/Providers/
```
