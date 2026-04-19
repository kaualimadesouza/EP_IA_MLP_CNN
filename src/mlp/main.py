"""Objetivo 1: MLP from scratch with Backpropagation."""

from pathlib import Path

import pandas as pd
from datasets import carregar_dados
from entities import MLP
from value_objects import DataChoiceEnum, Dataset


def salvar_erro_por_epoca(historico: list[float], caminho: str) -> None:
    """Salva o MSE de cada epoca em CSV (requisito da especificacao)."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enumerate(historico, 1), columns=["epoca", "mse"]).to_csv(
        caminho, index=False, float_format="%.6f"
    )


def main(
    data_choice: DataChoiceEnum = DataChoiceEnum.CARACTERES_COMPLETO,
    taxa_aprendizado: float = 0.1,
    epocas: int = 1000,
    num_neuronios_oculta: int = 10,
):
    # 1. Carregar os dados
    dataset: Dataset = carregar_dados(data_choice)

    # 2. Definir a arquitetura e criar a MLP
    arquitetura = [dataset.num_entradas, num_neuronios_oculta, dataset.num_saidas]
    mlp = MLP(arquitetura)

    print(f"Dataset: {data_choice.name}")
    print(f"Amostras: {len(dataset.treino)} treino / {len(dataset.teste)} teste")
    print(f"Arquitetura: {' -> '.join(str(n) for n in arquitetura)} (tanh)")
    print(f"Taxa de aprendizado: {taxa_aprendizado}")
    print(f"Epocas: {epocas}")
    print("\n--- Treinamento ---")

    # 3. Treinar a MLP
    historico = mlp.treinar(dataset.treino, taxa_aprendizado, epocas)

    # 3.1 Salvar o erro por epoca
    arquivo_erro = f"saidas/{data_choice.value.lower()}_erro_por_epoca.csv"
    salvar_erro_por_epoca(historico, arquivo_erro)

    print("\n--- Arquivos salvos ---")
    print(f"Erro por epoca: {arquivo_erro}")

    # 4. Testar a MLP
    print("\n--- Teste ---")
    resultados = mlp.testar(dataset.teste)
    acertos = sum(1 for r in resultados if r.acerto)
    total = len(resultados)
    print(f"Acuracia: {acertos}/{total} ({acertos / total:.2%})")


if __name__ == "__main__":
    main(
        data_choice=DataChoiceEnum.CARACTERES_COMPLETO,
        taxa_aprendizado=0.05,
        epocas=100000,
        num_neuronios_oculta=50,
    )
