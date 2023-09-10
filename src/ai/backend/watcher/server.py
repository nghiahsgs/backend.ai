import asyncio
import logging
import os
import signal
import ssl
import sys
from pprint import pformat, pprint
from typing import TYPE_CHECKING, Any, AsyncIterator

import aiohttp_cors
import aiotools
import click
from aiohttp import web

from ai.backend.common import config, utils
from ai.backend.common.config import redis_config_iv
from ai.backend.common.defs import REDIS_STREAM_DB
from ai.backend.common.etcd import AsyncEtcd, ConfigScopes
from ai.backend.common.events import (
    EventDispatcher,
    EventProducer,
)
from ai.backend.common.logging import BraceStyleAdapter
from ai.backend.common.types import LogSeverity

from . import __version__
from .config import watcher_config_iv
from .context import RootContext
from .defs import CORSOptions, WebMiddleware
from .plugin import WatcherPluginContext, WatcherWebAppPluginContext

if TYPE_CHECKING:
    from ai.backend.common.types import HostPortPair

log = BraceStyleAdapter(logging.getLogger(__spec__.name))  # type: ignore[name-defined]


async def ping(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "version": __version__,
        },
        status=200,
    )


def init_subapp(
    pkg_name: str,
    root_app: web.Application,
    subapp: web.Application,
    global_middlewares: list[WebMiddleware],
) -> None:
    async def _set_root_ctx(subapp: web.Application):
        # Allow subapp's access to the root app properties.
        # These are the public APIs exposed to plugins as well.
        subapp["ctx"] = root_app["ctx"]

    # We must copy the public interface prior to all user-defined startup signal handlers.
    subapp.on_startup.insert(0, _set_root_ctx)
    if "prefix" not in subapp:
        subapp["prefix"] = pkg_name.split(".")[-1].replace("_", "-")
    prefix = subapp["prefix"]
    root_app.add_subapp("/" + prefix, subapp)
    root_app.middlewares.extend(global_middlewares)


async def _init_subapp(
    root_app: web.Application,
    root_ctx: RootContext,
    etcd: AsyncEtcd,
    local_config: dict[str, Any],
    cors_options: CORSOptions,
) -> WatcherWebAppPluginContext:
    webapp_plugin_ctx = WatcherWebAppPluginContext(etcd, local_config)
    await webapp_plugin_ctx.init(
        root_ctx,
        allowlist=local_config["watcher"]["allowed-plugins"],
        blocklist=local_config["watcher"]["disabled-plugins"],
    )
    for plugin_name, plugin_instance in webapp_plugin_ctx.plugins.items():
        log.info(f"Loading webapp plugin: {plugin_name}")
        subapp, global_middlewares = await plugin_instance.create_app(cors_options)
        init_subapp(plugin_name, root_app, subapp, global_middlewares)

    return webapp_plugin_ctx


async def _init_watcher(
    ctx: RootContext,
    etcd: AsyncEtcd,
    local_config: dict[str, Any],
) -> WatcherPluginContext:
    watcher_ctx = WatcherPluginContext(etcd, local_config)
    module_config: dict[str, Any] = local_config["module"]
    await watcher_ctx.init()
    for plugin_name, plugin_instance in watcher_ctx.plugins.items():
        log.info("Loading watcher plugin: {0}", plugin_name)
        watcher_cls = plugin_instance.get_watcher_class()
        watcher_config_cls = watcher_cls.get_watcher_config_cls()
        watcher_config = watcher_config_cls.from_json(module_config[plugin_name])
        ctx.register_watcher(watcher_cls, watcher_config)
    return watcher_ctx


@aiotools.server
async def server_main(
    loop: asyncio.AbstractEventLoop,
    pidx: int,
    args: list[Any],
) -> AsyncIterator[None]:
    app = web.Application()
    cors_options = {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=False, expose_headers="*", allow_headers="*"
        ),
    }
    local_config = args[0]

    etcd_credentials = None
    if local_config["etcd"]["user"]:
        etcd_credentials = {
            "user": local_config["etcd"]["user"],
            "password": local_config["etcd"]["password"],
        }
    scope_prefix_map = {
        ConfigScopes.GLOBAL: "",
    }
    etcd = AsyncEtcd(
        local_config["etcd"]["addr"],
        local_config["etcd"]["namespace"],
        scope_prefix_map=scope_prefix_map,
        credentials=etcd_credentials,
    )
    app["config_server"] = etcd

    ssl_ctx = None
    if local_config["watcher"]["ssl-enabled"]:
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(
            str(local_config["watcher"]["ssl-cert"]),
            str(local_config["watcher"]["ssl-privkey"]),
        )
    redis_config = redis_config_iv.check(
        await etcd.get_prefix("config/redis"),
    )
    event_producer, event_dispatcher = None, None
    if local_config["watcher"]["event"]["connect-server"]:
        if (consumer_group := local_config["watcher"]["event"]["consumer-group"]) is None:
            raise RuntimeError("Should set valid `consumer-group` in local config file.")
        event_producer = await EventProducer.new(
            redis_config,
            db=REDIS_STREAM_DB,
            log_events=local_config["debug"]["log-events"],
        )
        event_dispatcher = await EventDispatcher.new(
            redis_config,
            db=REDIS_STREAM_DB,
            log_events=local_config["debug"]["log-events"],
            node_id=local_config["watcher"]["node-id"],
            consumer_group=consumer_group,
        )
    ctx = RootContext(
        pid=os.getpid(),
        node_id=local_config["watcher"]["node-id"],
        pidx=pidx,
        local_config=local_config,
        etcd=etcd,
        event_producer=event_producer,
        event_dispatcher=event_dispatcher,
    )
    app["ctx"] = ctx
    cors = aiohttp_cors.setup(app, defaults=cors_options)
    cors.add(app.router.add_route("GET", r"", ping))
    cors.add(app.router.add_route("GET", r"/", ping))

    watcher_ctx = await _init_watcher(ctx, etcd, local_config)
    webapp_plugin_ctx = await _init_subapp(app, ctx, etcd, local_config, cors_options)

    runner = web.AppRunner(app)
    await runner.setup()
    watcher_addr: HostPortPair = local_config["watcher"]["service-addr"]
    site = web.TCPSite(
        runner,
        str(watcher_addr.host),
        watcher_addr.port,
        backlog=5,
        reuse_port=True,
        ssl_context=ssl_ctx,
    )
    await site.start()
    log.info("started at {}", watcher_addr)
    try:
        yield
    finally:
        log.info("shutting down...")
        await webapp_plugin_ctx.cleanup()
        await watcher_ctx.cleanup()
        await runner.cleanup()


@click.command()
@click.argument("config_path", metavar="CONFIG")
@click.option(
    "--debug",
    is_flag=True,
    help=(
        "Alias of `--log-level debug`. It will override `--log-level` value to `debug` if this"
        " option is set."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(LogSeverity, case_sensitive=False),
    default=LogSeverity.INFO,
    help="Choose logging level from... debug, info, warning, error, critical",
)
@click.pass_context
def main(cli_ctx, config_path, log_level, debug=False):
    if debug:
        log_level = LogSeverity.DEBUG
    else:
        if (uid := os.geteuid()) != 0:
            raise RuntimeError(f"Watcher must be run as root, not {uid}. Abort.")

    raw_cfg, cfg_src_path = config.read_from_file(config_path, "watcher")

    config.override_with_env(raw_cfg, ("etcd", "namespace"), "BACKEND_NAMESPACE")
    config.override_with_env(raw_cfg, ("etcd", "addr"), "BACKEND_ETCD_ADDR")
    config.override_with_env(raw_cfg, ("etcd", "user"), "BACKEND_ETCD_USER")
    config.override_with_env(raw_cfg, ("etcd", "password"), "BACKEND_ETCD_PASSWORD")
    config.override_with_env(
        raw_cfg, ("watcher", "service-addr", "host"), "BACKEND_WATCHER_SERVICE_IP"
    )
    config.override_with_env(
        raw_cfg, ("watcher", "service-addr", "port"), "BACKEND_WATCHER_SERVICE_PORT"
    )
    if log_level == LogSeverity.DEBUG:
        config.override_key(raw_cfg, ("debug", "enabled"), True)

    try:
        cfg = config.check(raw_cfg, watcher_config_iv)
        if "debug" in cfg and cfg["debug"]["enabled"]:
            print("== Watcher configuration ==")
            pprint(cfg)
        cfg["_src"] = cfg_src_path
    except config.ConfigurationError as e:
        print("Validation of watcher configuration has failed:", file=sys.stderr)
        print(pformat(e.invalid_data), file=sys.stderr)
        raise click.Abort()

    # # Change the filename from the logging config's file section.
    # log_sockpath = Path(f"/tmp/backend.ai/ipc/watcher-logger-{os.getpid()}.sock")
    # log_sockpath.parent.mkdir(parents=True, exist_ok=True)
    # log_endpoint = f"ipc://{log_sockpath}"
    # cfg["logging"]["endpoint"] = log_endpoint
    # logger = Logger(cfg["logging"], is_master=True, log_endpoint=log_endpoint)
    # if "file" in cfg["logging"]["drivers"]:
    #     fn = Path(cfg["logging"]["file"]["filename"])
    #     cfg["logging"]["file"]["filename"] = f"{fn.stem}-watcher{fn.suffix}"

    # setproctitle(f"backend.ai: watcher {cfg['etcd']['namespace']}")
    # with logger:
    log.info("Backend.AI Watcher")
    log.info("runtime: {0}", utils.env_info())

    log_config = logging.getLogger("ai.backend.agent.config")
    log_config.debug("debug mode enabled.")

    aiotools.start_server(
        server_main,
        num_workers=1,
        args=(cfg,),
        stop_signals={signal.SIGINT, signal.SIGTERM, signal.SIGALRM},
    )
    log.info("exit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
