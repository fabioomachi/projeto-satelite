"""Módulo de captura RTSP com leitura seletiva — Projeto Satélite.

Arquitetura "Lentes":
    - Cada câmera Wi-Fi tem uma thread Daemon que drena o buffer de rede
      com ``cap.grab()`` (sem decodificar, custo quase zero de CPU).
    - Um "Sentinela" leve roda na CPU a 1-2 FPS: converte o frame para
      cinza em baixa resolução e subtrai de um FUNDO adaptativo
      (média móvel). Só há gasto de decode nesse ritmo baixo.
    - O pipeline principal (GPU/YOLO, futuro) só chama ``get_frame()``
      — que faz ``cap.retrieve()`` sob demanda — a ~5 FPS e apenas nas
      câmeras onde o Sentinela detectou movimento.
    - Se o Wi-Fi cair, a thread tenta reconectar a cada 5 s sem derrubar
      a aplicação principal.

Otimizando para: Ryzen 3 3200G + 16 GB RAM + GTX 1650 4 GB VRAM.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .motion import MotionSentinel

# Logger do módulo: o app principal pode configurar level/formato via logging.basicConfig.
logger = logging.getLogger(__name__)


@dataclass
class _CameraState:
    """Estado interno de uma única câmera RTSP.

    Attributes:
        camera_id: Identificador lógico (ex: "sala", "quintal").
        url: URL do stream RTSP.
        cap: Objeto VideoCapture do OpenCV (None quando desconectado).
        lock: Mutex que serializa grab() x retrieve() — VideoCapture
            NÃO é thread-safe, sem isso há condição de corrida/segfault.
        thread: Thread de drenagem dedicada a esta câmera.
        running: Flag de controle do loop infinito da thread.
        connected: True quando o stream está aberto e grab() funciona.
        motion_detected: Último resultado do Sentinela (com movimento?).
        sentinel: Detector de fundo adaptativo desta câmera.
        last_sentinel_ts: Timestamp da última análise do Sentinela.
        fail_count: Contador consecutivo de falhas (para log/backoff).
    """

    camera_id: str
    url: str
    cap: Optional[cv2.VideoCapture] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None
    running: bool = False
    connected: bool = False
    motion_detected: bool = False
    sentinel: Optional[MotionSentinel] = None
    last_sentinel_ts: float = 0.0
    fail_count: int = 0


class CameraManager:
    """Gerencia N streams RTSP com drenagem de buffer + sentinela de CPU.

    Example:
        >>> cams = [
        ...     {"id": "sala", "url": "rtsp://usuario:senha@192.168.0.101:554/stream"},
        ...     {"id": "quintal", "url": "rtsp://usuario:senha@192.168.0.102:554/stream"},
        ... ]
        >>> mgr = CameraManager(cameras=cams, sentinel_fps=1.0)
        >>> mgr.start()
        >>> # Loop principal: só decodifica a 5 FPS onde há movimento.
        >>> if mgr.has_motion("sala"):
        ...     frame = mgr.get_frame("sala")  # np.ndarray BGR ou None
        >>> mgr.stop()
    """

    def __init__(
        self,
        cameras: Union[List[Dict[str, Any]], Dict[str, str]],
        sentinel_fps: float = 1.0,
        sentinel_size: Tuple[int, int] = (160, 120),
        pixel_threshold: int = 25,
        motion_ratio: float = 0.02,
        sentinel_min_frames: int = 2,
        sentinel_cooldown_s: float = 3.0,
        sentinel_bg_alpha: float = 0.05,
        reconnect_delay: float = 5.0,
    ) -> None:
        """Inicializa o gerenciador (não conecta ainda — chame start()).

        Args:
            cameras: Lista de dicts ``[{"id": "...", "url": "rtsp://..."}]``.
                Também aceita ``{"id": "url"}`` por conveniência.
            sentinel_fps: Taxa do Sentinela de movimento (use 1.0 ou 2.0).
                Mantém o decode esporádico e a CPU livre.
            sentinel_size: Resolução interna do Sentinela (largura, altura).
                160x120 ≈ 19 mil pixels: absdiff aqui custa microssegundos.
            pixel_threshold: Limiar de intensidade (0-255) da diferença
                absoluta para considerar um pixel como "mudou".
            motion_ratio: Fração de pixels alterados (0.0-1.0) que dispara
                o flag de movimento. Ex: 0.02 = 2% da mini-imagem.
            sentinel_min_frames: Positivos seguidos para LIGAR o flag
                (rejeita spikes de ruído H.264).
            sentinel_cooldown_s: Segundos sem movimento até DESLIGAR o flag.
            sentinel_bg_alpha: Velocidade de aprendizado do fundo (0.0-1.0).
            reconnect_delay: Segundos entre tentativas de reconexão RTSP.
        """
        self._sentinel_fps: float = max(0.1, float(sentinel_fps))
        self._sentinel_size: Tuple[int, int] = sentinel_size
        self._pixel_threshold: int = int(pixel_threshold)
        self._motion_ratio: float = float(motion_ratio)
        self._sentinel_min_frames: int = max(1, int(sentinel_min_frames))
        self._sentinel_cooldown_s: float = max(0.0, float(sentinel_cooldown_s))
        self._sentinel_bg_alpha: float = float(sentinel_bg_alpha)
        self._reconnect_delay: float = max(1.0, float(reconnect_delay))

        # Normaliza a config para lista de (id, url), validando duplicatas.
        normalized: List[Tuple[str, str]] = self._normalize_config(cameras)

        self._cameras: Dict[str, _CameraState] = {}
        for cam_id, url in normalized:
            self._cameras[cam_id] = _CameraState(
                camera_id=cam_id,
                url=url,
                sentinel=MotionSentinel(
                    size=self._sentinel_size,
                    pixel_threshold=self._pixel_threshold,
                    motion_ratio=self._motion_ratio,
                    min_frames=self._sentinel_min_frames,
                    cooldown_s=self._sentinel_cooldown_s,
                    bg_alpha=self._sentinel_bg_alpha,
                ),
            )

        # Callbacks "acorde a IA": camera_id -> [fn(cam_id)]. Chamados na
        # transição False->True do Sentinela, na thread de drenagem.
        self._motion_callbacks: Dict[str, List[Callable[[str], None]]] = {
            cam_id: [] for cam_id, _ in normalized
        }

        self._started: bool = False

    # ------------------------------------------------------------------
    # Configuração
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_config(
        cameras: Union[List[Dict[str, Any]], Dict[str, str]],
    ) -> List[Tuple[str, str]]:
        """Converte a config (lista de dicts ou dict) em lista de tuplas.

        Raises:
            ValueError: Se a config estiver vazia, malformada ou com IDs duplicados.
        """
        items: List[Tuple[str, str]] = []

        if isinstance(cameras, dict):
            # Formato alternativo: {"sala": "rtsp://..."}
            for cam_id, url in cameras.items():
                items.append((str(cam_id), str(url)))
        elif isinstance(cameras, list):
            for entry in cameras:
                if not isinstance(entry, dict) or "id" not in entry or "url" not in entry:
                    raise ValueError(
                        "Cada câmera deve ser um dict com chaves 'id' e 'url'. "
                        f"Recebido: {entry!r}"
                    )
                items.append((str(entry["id"]), str(entry["url"])))
        else:
            raise ValueError("`cameras` deve ser lista de dicts ou dict id->url.")

        if not items:
            raise ValueError("Nenhuma câmera configurada.")

        vistos: set[str] = set()
        for cam_id, url in items:
            if not cam_id or not url:
                raise ValueError("`id` e `url` não podem ser vazios.")
            if cam_id in vistos:
                raise ValueError(f"ID de câmera duplicado: {cam_id!r}")
            vistos.add(cam_id)

        return items

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Abre as threads de drenagem de todas as câmeras (idempotente)."""
        if self._started:
            logger.warning("CameraManager já iniciado; ignorando start().")
            return

        self._started = True
        for state in self._cameras.values():
            state.running = True
            # Thread Daemon: morre com o processo principal e nunca trava o exit.
            t = threading.Thread(
                target=self._drain_loop,
                args=(state.camera_id,),
                name=f"dreno-{state.camera_id}",
                daemon=True,
            )
            state.thread = t
            t.start()
            logger.info("Thread de drenagem iniciada: %s", state.camera_id)

    def stop(self, timeout: float = 2.0) -> None:
        """Sinaliza parada, fecha os VideoCaptures e aguarda as threads.

        Args:
            timeout: Tempo máximo (s) para join de cada thread.
        """
        self._started = False
        for state in self._cameras.values():
            state.running = False

        for state in self._cameras.values():
            # Fecha o stream sob lock para não disputar com grab()/retrieve().
            try:
                with state.lock:
                    if state.cap is not None:
                        try:
                            state.cap.release()
                        except Exception as exc:  # noqa: BLE001 — liberar nunca pode travar o stop
                            logger.warning("Erro ao liberar %s: %s", state.camera_id, exc)
                        finally:
                            state.cap = None
                    state.connected = False
            except Exception as exc:  # noqa: BLE001 — robustez no shutdown
                logger.warning("Erro ao fechar %s: %s", state.camera_id, exc)

            if state.thread is not None and state.thread.is_alive():
                state.thread.join(timeout=timeout)

        logger.info("CameraManager parado.")

    # ------------------------------------------------------------------
    # Conexão RTSP (uso interno das threads)
    # ------------------------------------------------------------------
    def _connect(self, state: _CameraState) -> bool:
        """Tenta abrir o VideoCapture RTSP. Retorna True se conectou.

        Usa backend FFMPEG com buffer mínimo para reduzir latência.
        Nunca lança exceção — qualquer erro vira log + False.
        """
        try:
            with state.lock:
                # Garante que não sobrou um handle antigo meio-aberto.
                if state.cap is not None:
                    try:
                        state.cap.release()
                    except Exception:  # noqa: BLE001, S110 — melhor esforço
                        pass
                    state.cap = None

                cap = cv2.VideoCapture(state.url, cv2.CAP_FFMPEG)
                # Buffer 1: pede ao OpenCV/FFMPEG para não acumular frames
                # antigos (nossa thread de grab() já faz a drenagem fina).
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:  # noqa: BLE001 — nem todo backend suporta; ok ignorar
                    pass

                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001, S110 — melhor esforço
                        pass
                    return False

                state.cap = cap
                state.connected = True
                state.fail_count = 0
                # Zera o histórico do Sentinela: frame antigo geraria falso positivo.
                if state.sentinel is not None:
                    state.sentinel.reset()
                state.motion_detected = False
                return True
        except Exception as exc:  # noqa: BLE001 — rede/URL inválida não pode matar a thread
            logger.warning("Falha ao conectar %s: %s", state.camera_id, exc)
            state.connected = False
            return False

    # ------------------------------------------------------------------
    # Loop de drenagem + Sentinela (coração da economia de hardware)
    # ------------------------------------------------------------------
    def _drain_loop(self, camera_id: str) -> None:
        """Loop infinito da thread dedicada a UMA câmera.

        Lógica em 3 tempos, repetida até ``stop()``:
          1. Se desconectado → tenta ``_connect()``; se falhar, dorme
             ``reconnect_delay`` (5 s) e tenta de novo (resiliência Wi-Fi).
          2. Se conectado → ``cap.grab()``: só puxa o pacote da rede e
             descarta o buffer antigo SEM decodificar (barato p/ CPU).
          3. A cada 1/sentinel_fps → ``cap.retrieve()`` + análise leve
             na CPU (cinza, 160x120, absdiff). Só aqui há decode.
        """
        state = self._cameras[camera_id]
        sentinel_interval = 1.0 / self._sentinel_fps

        while state.running:
            try:
                # --- (1) Garante conexão; reconecta a cada 5 s se cair ---
                if state.cap is None or not state.connected:
                    ok = self._connect(state)
                    if not ok:
                        state.fail_count += 1
                        logger.warning(
                            "Stream %s indisponível (tentativa %d). "
                            "Nova tentativa em %.0f s.",
                            camera_id,
                            state.fail_count,
                            self._reconnect_delay,
                        )
                        # Sleep fatiado: responde rápido ao stop() mesmo
                        # durante a espera de reconexão.
                        self._sleep_interruptible(state, self._reconnect_delay)
                        continue
                    logger.info("Stream %s conectado.", camera_id)

                # --- (2) Drenagem do buffer: grab() sem decodificar ---
                grabbed = False
                try:
                    with state.lock:
                        if state.cap is not None:
                            grabbed = state.cap.grab()
                except Exception as exc:  # noqa: BLE001 — erro de rede/Camera IP
                    logger.warning("grab() falhou em %s: %s", camera_id, exc)
                    grabbed = False

                if not grabbed:
                    # Stream caiu no meio (Wi-Fi oscilou, câmera reiniciou...).
                    # Marca como desconectado; o topo do loop reconecta em 5 s.
                    logger.warning("Sinal perdido em %s; reconectando...", camera_id)
                    with state.lock:
                        if state.cap is not None:
                            try:
                                state.cap.release()
                            except Exception:  # noqa: BLE001, S110 — melhor esforço
                                pass
                            state.cap = None
                        state.connected = False
                        if state.sentinel is not None:
                            state.sentinel.reset()
                        state.motion_detected = False
                    self._sleep_interruptible(state, self._reconnect_delay)
                    continue

                # --- (3) Sentinela de movimento a 1-2 FPS (CPU) ---
                now = time.monotonic()
                if (now - state.last_sentinel_ts) >= sentinel_interval:
                    state.last_sentinel_ts = now
                    self._run_sentinel(state)

                # Micro-pausa: evita que o grab() em loop gire a 100% CPU.
                # O buffer continua fresco pois grab() é muito mais rápido que 30 FPS.
                time.sleep(0.01)

            except Exception as exc:  # noqa: BLE001 — a thread NUNCA pode morrer sozinha
                logger.exception("Erro inesperado no loop de %s: %s", camera_id, exc)
                self._sleep_interruptible(state, self._reconnect_delay)

        logger.info("Thread de drenagem encerrada: %s", camera_id)

    def _sleep_interruptible(self, state: _CameraState, seconds: float) -> None:
        """Dorme em fatias de 0,2 s para reagir rápido ao stop()."""
        deadline = time.monotonic() + seconds
        while state.running and time.monotonic() < deadline:
            time.sleep(min(0.2, deadline - time.monotonic()))

    # ------------------------------------------------------------------
    # Sentinela de movimento (estritamente CPU, imagem minúscula)
    # ------------------------------------------------------------------
    def _run_sentinel(self, state: _CameraState) -> None:
        """Decodifica UM frame e atualiza ``motion_detected`` via fundo adaptativo.

        Passos (todos na CPU, sem GPU/CUDA):
          1. ``retrieve()`` — único decode do ciclo do Sentinela.
          2. Converte para cinza e reduz para ~160x120 (poucos KB).
          3. Aplica blur leve p/ ignorar ruído/compressão do Wi-Fi.
          4. Delega ao ``MotionSentinel`` (subtração de fundo + debounce).
          5. Na transição False->True, dispara os callbacks ``on_motion``
             ("Tem alguém na sala, ligue a IA").
        """
        try:
            with state.lock:
                if state.cap is None:
                    return
                ok, frame = state.cap.retrieve()
            if not ok or frame is None:
                return

            # Cinza + baixa resolução: de 1080p (~6 MB) para ~19 KB.
            # INTER_AREA é o ideal para downscale (evita serrilhado).
            small = cv2.resize(
                frame, self._sentinel_size, interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # Blur 5x5 remove chuvisco do H.264/Wi-Fi sem apagar gente andando.
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if state.sentinel is None:
                return
            was = state.motion_detected
            state.motion_detected = state.sentinel.update(gray)

            if state.motion_detected and not was:
                self._fire_motion_callbacks(state.camera_id)

        except Exception as exc:  # noqa: BLE001 — Sentinela jamais derruba a thread
            logger.warning("Sentinela falhou em %s: %s", state.camera_id, exc)

    def _fire_motion_callbacks(self, camera_id: str) -> None:
        """Invoca os callbacks de movimento (um nunca derruba os outros)."""
        for cb in list(self._motion_callbacks.get(camera_id, [])):
            try:
                cb(camera_id)
            except Exception as exc:  # noqa: BLE001 — callback ruim não mata a thread
                logger.warning("Callback on_motion(%s) falhou: %s", camera_id, exc)

    # ------------------------------------------------------------------
    # API pública de leitura
    # ------------------------------------------------------------------
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Recupera o frame mais recente da câmera (decode sob demanda).

        Executa APENAS ``cap.retrieve()`` — o frame retornado corresponde
        ao último ``grab()`` da thread de drenagem, ou seja, tem no máximo
        algumas dezenas de ms de idade (sem delay acumulado).

        O pipeline principal deve chamar isso a ~5 FPS e SOMENTE quando
        ``has_motion(camera_id)`` for True, para poupar CPU/GPU/VRAM.

        Args:
            camera_id: ID informado no construtor.

        Returns:
            Frame BGR (np.ndarray) ou None se desconectado/sem frame.

        Raises:
            KeyError: Se o camera_id não existir.
        """
        state = self._get_state(camera_id)
        try:
            with state.lock:
                if state.cap is None or not state.connected:
                    return None
                ok, frame = state.cap.retrieve()
            if not ok or frame is None:
                return None
            return frame
        except KeyError:
            raise
        except Exception as exc:  # noqa: BLE001 — erro de rede vira None, não exceção
            logger.warning("get_frame(%s) falhou: %s", camera_id, exc)
            return None

    def has_motion(self, camera_id: str) -> bool:
        """Retorna True se o Sentinela viu movimento na câmera.

        Raises:
            KeyError: Se o camera_id não existir.
        """
        return self._get_state(camera_id).motion_detected

    def on_motion(
        self, camera_id: str, callback: Callable[[str], None]
    ) -> None:
        """Registra um callback disparado quando o Sentinela LIGA o movimento.

        É o gancho "acorde a IA": o pipeline GPU/YOLO futuro assina aqui e
        só roda inferência nas câmeras/trechos com gente de verdade.

        O callback recebe o ``camera_id`` e roda na thread de drenagem —
        mantenha-o RÁPIDO (só sinalize; processe pesado em outra thread).

        Raises:
            KeyError: Se o camera_id não existir.
        """
        self._get_state(camera_id)  # valida o ID (lança KeyError amigável)
        self._motion_callbacks[camera_id].append(callback)

    def get_motion_snapshot(self) -> Dict[str, bool]:
        """Foto instantânea do movimento de TODAS as câmeras."""
        return {cid: st.motion_detected for cid, st in self._cameras.items()}

    def is_connected(self, camera_id: str) -> bool:
        """Retorna True se o stream RTSP está aberto.

        Raises:
            KeyError: Se o camera_id não existir.
        """
        return self._get_state(camera_id).connected

    def list_cameras(self) -> List[str]:
        """Lista os IDs de câmera gerenciados."""
        return list(self._cameras.keys())

    # ------------------------------------------------------------------
    # Util interno
    # ------------------------------------------------------------------
    def _get_state(self, camera_id: str) -> _CameraState:
        """Busca o estado da câmera ou lança KeyError amigável."""
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise KeyError(
                f"Câmera desconhecida: {camera_id!r}. "
                f"Disponíveis: {sorted(self._cameras)}"
            ) from None
