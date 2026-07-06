"""FastAPI app factory + route registration.

`create_app(paths)` returns a fully-wired FastAPI instance. Callers
(both `rabbit serve` and the pytest suite) construct with a Paths
object so the same code drives production and isolated tests.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .paths import Paths
from .routes import (
    analytics as analytics_routes,
    backtests as backtests_routes,
    config as config_routes,
    data as data_routes,
    models as models_routes,
    shadow as shadow_routes,
)


STATIC_DIR = Path(__file__).parent / "static"


def create_app(paths: Paths | None = None) -> FastAPI:
    paths = paths or Paths()
    app = FastAPI(
        title="Rabbit Hunter",
        description="Quantitative crypto perpetual engine — operator UI",
        version="0.2.0",
        docs_url="/docs",
    )
    # Stash paths so routes can grab it via request.app.state
    app.state.paths = paths

    # Register domain routers
    app.include_router(shadow_routes.router, prefix="/api/shadow", tags=["shadow"])
    app.include_router(backtests_routes.router, prefix="/api/backtests",
                        tags=["backtests"])
    app.include_router(models_routes.router, prefix="/api/models", tags=["models"])
    app.include_router(config_routes.router, prefix="/api/config", tags=["config"])
    app.include_router(analytics_routes.router, prefix="/api/analytics",
                        tags=["analytics"])
    app.include_router(data_routes.router, prefix="/api/data", tags=["data"])

    # Serve the SPA index at "/" and any known bundled static asset.
    @app.get("/", include_in_schema=False)
    def _index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", include_in_schema=False)
    def _health():
        return {"ok": True, "root": str(paths.root)}

    return app
