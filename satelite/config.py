"""Configuração do Projeto Satélite via arquivo `.env`.

Carrega as variáveis do `.env` (raiz do repositório) com `python-dotenv`
e expõe helpers para montar URLs RTSP e listar as câmeras configuradas.

Precedência (da maior para a menor):
    1. Variáveis de ambiente reais do SO.
    2. Valores do arquivo `.env`.
    3. Defaults não-sensíveis (porta, canal) — IP/usuário/senha NÃO têm
       default: se faltarem, `load_cameras()` lança `ValueError` explícito.

Esquema multi-câmera (ver `.env.example`)::

    CAM_COUNT=2
    CAM_1_ID=cam1
    CAM_1_IP=192.168.x.x
    ...
    CAM_2_ID=cam2
    ...
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Carrega o `.env` da raiz do repo (ao lado de `main.py`). `override=False`
# garante que env vars reais do SO sempre vençam o arquivo.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

#: Transporte FFmpeg padrão. A Câmera 1 (HIipCamera, H.265) só funciona via
#: UDP ("Nonmatching transport in server reply" com TCP) — verificado em campo.
DEFAULT_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;udp"


def get_ffmpeg_capture_options() -> str:
    """Retorna as opções de captura FFmpeg (`OPENCV_FFMPEG_CAPTURE_OPTIONS`)."""
    return os.getenv(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS", DEFAULT_FFMPEG_CAPTURE_OPTIONS
    )


def build_rtsp_url(
    ip: str,
    port: str | int,
    user: str,
    password: str,
    channel: str = "/onvif1",
) -> str:
    """Monta a URL RTSP codificando a senha com `quote()` (`#` -> `%23`).

    Args:
        ip: IP ou hostname da câmera.
        port: Porta RTSP (padrão 554).
        user: Usuário do stream.
        password: Senha **em texto claro** (é codificada aqui dentro).
        channel: Caminho do stream (ex: `/onvif1`).

    Returns:
        URL pronta para o `CameraManager`.
    """
    safe_password = quote(str(password), safe="")
    if not channel.startswith("/"):
        channel = f"/{channel}"
    return f"rtsp://{user}:{safe_password}@{ip}:{port}{channel}"


def _required(var: str, cam_label: str) -> str:
    """Lê uma env var obrigatória ou lança `ValueError` identificando-a."""
    value = os.getenv(var, "").strip()
    if not value:
        raise ValueError(
            f"Configuração ausente: {var} ({cam_label}). "
            f"Defina no `.env` (veja `.env.example`) ou como variável de ambiente."
        )
    return value


def _get_float(var: str, default: float) -> float:
    """Lê um float do ambiente ou retorna o default (com erro explícito)."""
    raw = os.getenv(var, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"{var} inválido: {raw!r}. Use um número (default {default})."
        ) from None


def _get_int(var: str, default: int) -> int:
    """Lê um int do ambiente ou retorna o default (com erro explícito)."""
    raw = os.getenv(var, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"{var} inválido: {raw!r}. Use um inteiro (default {default})."
        ) from None


def load_sentinel_params() -> Dict[str, Any]:
    """Lê os parâmetros do Sentinela (`SENTINEL_*`) do `.env`/ambiente.

    Returns:
        Dict pronto para ``CameraManager(..., **params)`` — chaves
        ``sentinel_fps/size``, ``pixel_threshold``, ``motion_ratio``,
        ``sentinel_min_frames/cooldown_s/bg_alpha``.
    """
    width = _get_int("SENTINEL_WIDTH", 160)
    height = _get_int("SENTINEL_HEIGHT", 120)
    return {
        "sentinel_fps": _get_float("SENTINEL_FPS", 1.0),
        "sentinel_size": (width, height),
        "pixel_threshold": _get_int("SENTINEL_THRESHOLD", 25),
        "motion_ratio": _get_float("SENTINEL_MOTION_RATIO", 0.02),
        "sentinel_min_frames": _get_int("SENTINEL_MIN_FRAMES", 2),
        "sentinel_cooldown_s": _get_float("SENTINEL_COOLDOWN_S", 3.0),
        "sentinel_bg_alpha": _get_float("SENTINEL_BG_ALPHA", 0.05),
    }


def load_cameras() -> List[Dict[str, Any]]:
    """Lê o `.env` e retorna a lista de câmeras no formato do `CameraManager`.

    Returns:
        Lista de dicts ``[{"id": "...", "url": "rtsp://..."}]``.

    Raises:
        ValueError: Se `CAM_COUNT` for inválido, alguma var obrigatória
            (`CAM_{n}_IP/USER/PASSWORD`) faltar ou houver IDs duplicados.
    """
    try:
        count = int(os.getenv("CAM_COUNT", "1"))
    except ValueError:
        raise ValueError(
            f"CAM_COUNT inválido: {os.getenv('CAM_COUNT')!r}. Use um inteiro >= 1."
        ) from None
    if count < 1:
        raise ValueError(f"CAM_COUNT inválido: {count}. Use um inteiro >= 1.")

    cameras: List[Dict[str, Any]] = []
    vistos: set[str] = set()
    for n in range(1, count + 1):
        p = f"CAM_{n}"
        label = f"câmera {n} ({p}_*)"
        cam_id = os.getenv(f"{p}_ID", f"cam{n}").strip() or f"cam{n}"
        if cam_id in vistos:
            raise ValueError(f"ID de câmera duplicado: {cam_id!r}.")
        vistos.add(cam_id)

        ip = _required(f"{p}_IP", label)
        user = _required(f"{p}_USER", label)
        password = _required(f"{p}_PASSWORD", label)
        port = os.getenv(f"{p}_PORT", "554").strip() or "554"
        channel = os.getenv(f"{p}_CHANNEL", "/onvif1").strip() or "/onvif1"

        url = build_rtsp_url(ip, port, user, password, channel)
        cameras.append({"id": cam_id, "url": url})
        logger.info("Câmera %s configurada: rtsp://%s:***@%s:%s%s (senha omitida)",
                    cam_id, user, ip, port,
                    channel if channel.startswith("/") else f"/{channel}")

    return cameras
