"""Carregamento dos datasets usados no EP (portas logicas e caracteres)."""

import numpy as np
import pandas as pd
from value_objects import Amostra, DataChoiceEnum, Dataset

PATHS_DATASETS: dict[DataChoiceEnum, str] = {
    DataChoiceEnum.OR: "data/portas_logicas/or.csv",
    DataChoiceEnum.AND: "data/portas_logicas/and.csv",
    DataChoiceEnum.XOR: "data/portas_logicas/xor.csv",
    DataChoiceEnum.CARACTERES_REDUZIDO: "data/caracteres_reduzido",
    DataChoiceEnum.CARACTERES_COMPLETO: "data/caracteres_completo",
}


def _df_para_amostras(df: pd.DataFrame, num_entradas: int) -> list[Amostra]:
    amostras: list[Amostra] = []
    for _, linha in df.iterrows():
        entrada = linha.iloc[:num_entradas].tolist()
        esperado = linha.iloc[num_entradas:].tolist()
        amostra = Amostra(entrada=entrada, esperado=esperado)
        amostras.append(amostra)
    return amostras


def _carregar_porta_logica(path: str) -> Dataset:
    df = pd.read_csv(path, header=None)

    # O número de entradas é o número de colunas menos 1 (a última é a saída).
    num_entradas = len(df.columns) - 1

    # As mesmas 4 amostras servem de treino e teste, mas guardamos
    # em listas diferentes pra evitar aliasing (ex.: se treinar() embaralhar
    # in-place, o conjunto de teste nao deve ser afetado).
    treino = _df_para_amostras(df, num_entradas)
    teste = _df_para_amostras(df, num_entradas)

    return Dataset(
        treino=treino,
        teste=teste,
        num_entradas=num_entradas,
        num_saidas=1,
    )


def _carregar_caracteres_reduzido(pasta: str) -> Dataset:
    limpo = pd.read_csv(f"{pasta}/limpo.csv", header=None)
    ruido = pd.read_csv(f"{pasta}/ruido.csv", header=None)
    ruido20 = pd.read_csv(f"{pasta}/ruido20.csv", header=None)

    num_entradas = max(len(limpo.columns), len(ruido.columns), len(ruido20.columns)) - 7

    treino = _df_para_amostras(limpo, num_entradas)
    teste = _df_para_amostras(ruido, num_entradas) + _df_para_amostras(
        ruido20, num_entradas
    )
    return Dataset(treino=treino, teste=teste, num_entradas=num_entradas, num_saidas=7)


def _carregar_caracteres_completo(pasta: str) -> Dataset:
    # X.npy vem em formato de imagem (N, 10, 12, 1) -> achatamos pra MLP.
    X = np.load(f"{pasta}/X.npy").reshape(-1, 120)
    Y = np.load(f"{pasta}/Y_classe.npy")

    amostras: list[Amostra] = []
    for x, y in zip(X, Y, strict=True):
        amostras.append(Amostra(entrada=x.tolist(), esperado=y.tolist()))

    # Split 80/20 com seed fixa pra reproducibilidade.
    corte = len(amostras) * 80 // 100
    return Dataset(
        treino=amostras[:corte],
        teste=amostras[corte:],
        num_entradas=X.shape[1],
        num_saidas=Y.shape[1],
    )


def carregar_dados(data_choice: DataChoiceEnum) -> Dataset:
    path = PATHS_DATASETS[data_choice]
    match data_choice:
        case DataChoiceEnum.OR | DataChoiceEnum.AND | DataChoiceEnum.XOR:
            return _carregar_porta_logica(path)
        case DataChoiceEnum.CARACTERES_REDUZIDO:
            return _carregar_caracteres_reduzido(path)
        case DataChoiceEnum.CARACTERES_COMPLETO:
            return _carregar_caracteres_completo(path)
        case _:
            raise ValueError(f"Escolha de dataset inválida: {data_choice!r}")
