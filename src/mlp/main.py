"""Objetivo 1: MLP from scratch with Backpropagation."""

from entities import MLP
from value_objects import DataChoiceEnum, Dataset


def carregar_dados(data_choice: DataChoiceEnum) -> Dataset:
    # TODO: Implementar a lógica para carregar os dados com base na escolha
    pass


def main(
    data_choice: DataChoiceEnum = DataChoiceEnum.CARACTERES_COMPLETO,
    taxa_aprendizado: float = 0.1,
    epocas: int = 1000,
    num_neuronios_oculta: int = 10,
):
    # TODO: 1. Carregar os dados
    dataset: Dataset = carregar_dados(data_choice)

    # 2. Definir a arquitetura da MLP (entradas, ocultas, saidas)
    camadas = [dataset.num_entradas, num_neuronios_oculta, dataset.num_saidas]
    mlp: MLP = MLP(camadas)

    # TODO: 3. Treinar a MLP (forward + backpropagation)
    mlp.treinar(dataset.treino, taxa_aprendizado, epocas)
    # TODO: 4. Testar a MLP e avaliar resultados


if __name__ == "__main__":
    main()
