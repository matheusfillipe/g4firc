from typing import Any, Generator, Union

from g4f import models
from g4f.Provider import BaseProvider
from trio_asyncio import aio_as_trio

CreateResult = Generator[str, None, None]


@aio_as_trio
async def create(
    model: Union[models.Model, str],
    messages: list[dict[str, str]],
    provider: Union[type[BaseProvider], None] = None,
    stream: bool = False,
    auth: Union[str, None] = None,
    **kwargs: Any,
) -> Union[CreateResult, str]:
    if isinstance(model, str):
        try:
            model = models.ModelUtils.convert[model]
        except KeyError:
            raise Exception(f"The model: {model} does not exist")

    provider = model.best_provider if provider == None else provider

    if not provider.working:
        raise Exception(f"{provider.__name__} is not working")

    if provider.needs_auth and not auth:
        raise Exception(
            f"ValueError: {provider.__name__} requires authentication (use auth='cookie or token or jwt ...' param)"
        )

    if provider.needs_auth:
        kwargs["auth"] = auth

    if not provider.supports_stream and stream:
        raise Exception(f'ValueError: {provider.__name__} does not support "stream" argument')

    print(f"Using {provider.__name__} provider")

    if hasattr(provider, "create_async"):
        return await provider.create_async(model.name, messages, **kwargs)

    result = provider.create_completion(model.name, messages, stream, **kwargs)
    return result if stream else "".join(result)
