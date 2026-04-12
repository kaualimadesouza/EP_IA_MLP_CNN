# EP IA: MLP + CNN

Trabalho da disciplina **Inteligencia Artificial (ACH2016)**, USP EACH, 1o semestre de 2026.

## Objetivos

1. **MLP (Multilayer Perceptron)**, Implementacao do zero (sem frameworks de redes neurais), treinada com Backpropagation.
   - Datasets: portas logicas (OR, AND, XOR), CARACTERES, CARACTERES COMPLETO

2. **CNN (Convolutional Neural Network)**, Implementacao com framework, testada no Fashion MNIST.

## Estrutura

```
EP_IA_MLP_CNN/
├── src/
│   ├── mlp/           # Objetivo 1: MLP from scratch
│   └── cnn/           # Objetivo 2: CNN com framework
├── data/              # Datasets
├── outputs/           # Pesos, erros, hiperparametros
├── docs/              # Enunciado e documentacao
├── pyproject.toml
└── README.md
```

## Por que uv ao inves de pip?

Neste projeto usamos o [uv](https://docs.astral.sh/uv/) como gerenciador de pacotes. Ele substitui pip, venv e pip-tools num unico comando.

**Diferenca pratica:**

Com pip (forma tradicional):
```bash
python -m venv .venv            # cria o ambiente virtual
source .venv/bin/activate       # ativa o ambiente
pip install -r requirements.txt # instala dependencias
```

Com uv:
```bash
uv sync                         # faz tudo de uma vez
```

`uv sync` le o `pyproject.toml`, cria o `.venv` automaticamente, resolve versoes e instala tudo. Nao precisa de `requirements.txt`, as dependencias ficam no `pyproject.toml`.

Para rodar scripts dentro do ambiente:
```bash
uv run python src/mlp/main.py   # roda com o .venv ativo automaticamente
```

Ou ative o venv manualmente e use `python` direto:
```bash
source .venv/bin/activate
python src/mlp/main.py
```

## Setup

```bash
make install
```

Isso instala o uv (se necessario), as dependencias e configura os hooks de pre-commit.

## Uso

```bash
source .venv/bin/activate
python src/mlp/main.py
python src/cnn/main.py
```

## Entrega

- Data limite: **2 de junho de 2026, 23h55**
- Via e-Disciplinas
