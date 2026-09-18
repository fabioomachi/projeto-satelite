"""Ponto de entrada principal — Projeto Satélite (módulo Lentes).

Inicializa e testa o `satelite/lentes/camera_manager.py` com as câmeras
configuradas no arquivo `.env` (veja `.env.example`).

Arquitetura de atenção: o Sentinela (CPU, 1-2 FPS, fundo adaptativo)
vigia todas as câmeras e o pipeline pesado (decode 1080p, futura YOLO/GPU)
só roda onde há movimento — a GPU "dorme" até o Sentinela avisar.

Uso:
    cp .env.example .env   # uma vez; preencha com os dados reais
    python main.py

Variáveis de ambiente reais do SO têm precedência sobre o `.env`.

Interrompa com Ctrl+C — as threads do CameraManager são encerradas
de forma limpa via `manager.stop()`.
"""

import logging
import os

from satelite.config import get_ffmpeg_capture_options

# Transporte RTSP (Linux). Precisa ser definido ANTES de importar o cv2
# (importado dentro de camera_manager). Vem do `.env`, sem segredos no código.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = get_ffmpeg_capture_options()

import time

from satelite.config import load_cameras, load_sentinel_params
from satelite.lentes.camera_manager import CameraManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def on_motion_wake_ai(camera_id: str) -> None:
    """Callback "acorde a IA": o Sentinela viu gente na câmera.

    Placeholder onde o pipeline GPU/YOLO vai entrar — por enquanto,
    só registra o evento. Mantido RÁPIDO: roda na thread de drenagem.
    """
    logger.warning(">>> IA ACORDADA em [%s]: movimento detectado!", camera_id)


def main() -> None:
    cameras = load_cameras()
    sentinel_params = load_sentinel_params()

    # Instancia o gerenciador já com as câmeras configuradas...
    manager = CameraManager(cameras=cameras, **sentinel_params)
    # ...assina o "acorde a IA" e inicia as threads de fundo.
    for cam in cameras:
        manager.on_motion(cam["id"], on_motion_wake_ai)
    manager.start()
    logger.info("CameraManager iniciado. Pressione Ctrl+C para encerrar.")

    cam_ids = manager.list_cameras()
    try:
        while True:
            loop_start = time.monotonic()

            for cam_id in cam_ids:
                conectado = manager.is_connected(cam_id)
                movimento = manager.has_motion(cam_id)

                # GATE DE ATENÇÃO: sem movimento, a IA dorme — pula o
                # decode 1080p (caro) e só mostra o estado do Sentinela.
                if not movimento:
                    print(f"[{cam_id}] DORMINDO | movimento=False | "
                          f"conectado={conectado} (IA off)")
                    continue

                frame = manager.get_frame(cam_id)
                if frame is not None:
                    h, w = frame.shape[:2]
                    print(
                        f"[{cam_id}] IA ON | resolucao={w}x{h} | "
                        f"movimento=True | conectado={conectado}"
                    )
                else:
                    print(
                        f"[{cam_id}] SEM FRAME | movimento=True | "
                        f"conectado={conectado} (tentando reconectar...)"
                    )

            # Vigia a 1 segundo (desconta o tempo de get_frame).
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, 1.0 - elapsed))

    except KeyboardInterrupt:
        print("\nCtrl+C recebido — encerrando...")
    finally:
        manager.stop()
        logger.info("CameraManager parado. Até logo!")


if __name__ == "__main__":
    main()
