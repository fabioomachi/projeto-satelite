"""Sentinela de movimento ultraleve — Projeto Satélite.

Subtração de fundo matemática, 100% CPU, sem GPU/CUDA:

    1. Mantém um "fundo" adaptativo (média móvel da cena em 160x120).
    2. Compara cada amostra com o fundo via diferença absoluta.
    3. Dispara movimento quando a fração de pixels alterados passa do limiar.
    4. Debounce: exige N positivos seguidos para LIGAR e um cooldown
       sem movimento para DESLIGAR (ignora flicker de ruído H.264/chuva).

O fundo só aprende quando NÃO há movimento — uma pessoa parada na cena
não "vira fundo". Mudanças lentas de luz (sol/nuvem) são absorvidas.

Custo típico: ~19 mil pixels x poucas operações por amostra, a 1-2 FPS:
microssegundos no Ryzen 3.

Example:
    >>> sentinel = MotionSentinel()
    >>> if sentinel.update(mini_gray):  # np.ndarray uint8, 160x120
    ...     print("movimento!")
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class MotionSentinel:
    """Detector de movimento por fundo adaptativo para UMA câmera."""

    def __init__(
        self,
        size: tuple[int, int] = (160, 120),
        pixel_threshold: int = 25,
        motion_ratio: float = 0.02,
        min_frames: int = 2,
        cooldown_s: float = 3.0,
        bg_alpha: float = 0.05,
    ) -> None:
        """Inicializa o Sentinela (sem histórico — o fundo aprende sozinho).

        Args:
            size: Resolução interna (largura, altura). 160x120 ≈ 19 mil
                pixels: absdiff aqui custa microssegundos.
            pixel_threshold: Limiar de intensidade (0-255) da diferença
                absoluta para considerar um pixel como "mudou".
            motion_ratio: Fração de pixels alterados (0.0-1.0) que caracteriza
                movimento. Ex: 0.02 = 2% da mini-imagem.
            min_frames: Positivos seguidos necessários para LIGAR o flag
                (rejeita spikes isolados de ruído/compressão).
            cooldown_s: Segundos sem movimento até DESLIGAR o flag.
            bg_alpha: Velocidade de aprendizado do fundo (0.0-1.0).
                0.05 absorve mudanças lentas de luz sem engolir gente.
        """
        self.size = (int(size[0]), int(size[1]))
        self.pixel_threshold = int(pixel_threshold)
        self.motion_ratio = float(motion_ratio)
        self.min_frames = max(1, int(min_frames))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.bg_alpha = min(1.0, max(0.0, float(bg_alpha)))

        self._bg: np.ndarray | None = None  # fundo float32
        self._hits: int = 0  # positivos seguidos
        self._last_raw_ts: float = 0.0  # último positivo (qualquer um)
        self._motion: bool = False
        self.last_ratio: float = 0.0  # diagnóstico: ratio da última amostra

    @property
    def motion_detected(self) -> bool:
        """Flag com debounce: True = movimento sustentado na cena."""
        return self._motion

    def reset(self) -> None:
        """Descarta o fundo aprendido (use ao reconectar o stream)."""
        self._bg = None
        self._hits = 0
        self._motion = False
        self.last_ratio = 0.0

    def update(
        self, gray: np.ndarray, now: float | None = None
    ) -> bool:
        """Processa UMA amostra e retorna o flag de movimento com debounce.

        Args:
            gray: Imagem cinza uint8 na resolução `size` (já com blur leve
                aplicado pelo chamador, se desejado).
            now: Timestamp (para testes). Padrão: `time.monotonic()`.

        Returns:
            True se há movimento sustentado, False caso contrário.
        """
        ts = time.monotonic() if now is None else float(now)

        if (
            self._bg is None
            or self._bg.shape[:2] != gray.shape[:2]
        ):
            # Primeira amostra (ou resolução mudou): aprende o fundo,
            # sem referência ainda — assume sem movimento.
            self._bg = gray.astype(np.float32)
            self._hits = 0
            self._motion = False
            self.last_ratio = 0.0
            return False

        # Diferença absoluta contra o FUNDO (não contra o frame anterior).
        bg_u8 = cv2.convertScaleAbs(self._bg)
        diff = cv2.absdiff(bg_u8, gray)
        _, thresh = cv2.threshold(
            diff, self.pixel_threshold, 255, cv2.THRESH_BINARY
        )
        changed = int(cv2.countNonZero(thresh))
        ratio = changed / max(1, thresh.size)
        self.last_ratio = ratio

        raw = ratio >= self.motion_ratio
        if raw:
            self._hits += 1
            self._last_raw_ts = ts
            # Fundo NÃO aprende com movimento: pessoa parada continua
            # diferente do fundo e segue detectável.
            if self._hits >= self.min_frames:
                self._motion = True
        else:
            self._hits = 0
            # Aprende devagar: absorve sol/nuvem, ignora gente passando.
            cv2.accumulateWeighted(gray, self._bg, self.bg_alpha)
            if self._motion and (ts - self._last_raw_ts) >= self.cooldown_s:
                self._motion = False

        return self._motion
