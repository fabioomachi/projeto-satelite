# 🛰️ Projeto Satélite

**Monitoramento Residencial Autônomo com Visão Computacional para Cuidado e Bem-Estar**

O **Projeto Satélite** é um sistema de visão computacional 100% local e focado em privacidade. Mais do que uma ferramenta de segurança, ele é uma rede de apoio tecnológico que orquestra múltiplas câmeras RTSP para observar o perímetro, mapear rotinas e ajudar a cuidar do bem-estar emocional e físico de uma família.

## ❤️ A Motivação

Este projeto nasceu de uma necessidade profundamente pessoal. Após a perda da minha esposa, Edlaine, me vi diante do imenso desafio de gerenciar o luto e a rotina da casa, cuidando de mim e da minha filha. No meio da dor, nem sempre temos a clareza para perceber nossos próprios sinais de sofrimento, isolamento ou mudanças bruscas de comportamento. 

O *Satélite* foi criado para ser esse olhar atento e silencioso. Ele não julga, apenas observa a nossa rotina de forma totalmente privada. Seu objetivo é me ajudar a monitorar nosso bem-estar diário, classificando ações e emitindo alertas gentis caso detecte padrões fora do normal, garantindo que eu possa agir e cuidar de nós com mais precisão.

## 🎯 O Desafio Técnico e a Solução
Processar múltiplos streams de vídeo simultâneos com IA geralmente exige GPUs de altíssimo custo. O diferencial do *Satélite* é a sua **arquitetura de atenção**. 

Desenvolvido e otimizado para rodar em hardware de entrada (AMD Ryzen 3, 16GB RAM e uma GTX 1650 de 4GB), o sistema atua como um "diretor". A CPU faz a triagem leve de movimento, e a GPU (via CUDA) só é acionada para analisar a câmera onde a ação realmente está acontecendo, reduzindo drasticamente o consumo de energia e memória.

## 🚀 Principais Funcionalidades
* **Mecanismo de Atenção RTSP:** Leitura de frames inteligente que evita atrasos de vídeo e economiza VRAM.
* **Handover Multi-Câmera (Re-Identificação):** Rastreia e mantém a identidade (ID) das pessoas enquanto transitam perfeitamente entre os cômodos da casa.
* **Detecção de Pose e Comportamento:** Combina YOLOv8 (GPU) para detecção humana e MediaPipe (CPU) para mapeamento do esqueleto e da postura.
* **Active Learning via Telegram:** Em caso de movimentos anômalos ou com baixa confiança matemática (<90%), o sistema recorta um pequeno clipe e envia para o meu Telegram de forma privada. Lá, eu utilizo botões interativos para classificar a ação ("Taggear") e ensinar o modelo localmente sobre o que é normal ou o que requer atenção.

## 📁 Estrutura da Base
A arquitetura foi dividida fazendo alusão a componentes espaciais para manter a modularidade:
* `/satelite/lentes/`: Conexão RTSP, CameraManager e leitura seletiva.
* `/satelite/radar/`: Motor de Re-Identificação (Tracking/Handover).
* `/satelite/orbita/`: Lógica de zonas, geometria e mapeamento dos cômodos.
* `/satelite/processamento/`: Pipeline neural (YOLO e MediaPipe).
* `/satelite/telemetria/`: Integração com Telegram Bot, alertas e banco SQLite.

## ⚖️ Licença e Uso
Este projeto adota a licença **PolyForm Noncommercial 1.0.0**.
O código é aberto para que outras famílias e pessoas que precisem de redes de apoio semelhantes possam baixar, estudar, modificar e instalar em suas casas livremente. No entanto, é **estritamente proibido** empacotar este código para uso comercial, venda ou oferta como serviço (SaaS). O cuidado não deve ser monetizado por terceiros.

## ☕ Como Apoiar (Sponsor)
O *Satélite* é mantido de forma independente. Se esta iniciativa tocou você, ajudou a automatizar a sua casa ou a aprender mais sobre visão computacional focada no cuidado humano, considere apoiar o desenvolvimento através do botão **Sponsor** no topo do repositório. Toda contribuição ajuda a manter este projeto vivo.