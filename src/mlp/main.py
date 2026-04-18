"""Objetivo 1: MLP from scratch with Backpropagation."""

from datasets import carregar_dados
from entities import MLP
from value_objects import DataChoiceEnum, Dataset


def main(
    data_choice: DataChoiceEnum = DataChoiceEnum.CARACTERES_COMPLETO,
    taxa_aprendizado: float = 0.1,
    epocas: int = 1000,
    num_neuronios_oculta: int = 10,
):
    # 1.1 Carregar os dados (dataset de treino e teste)
    dataset: Dataset = carregar_dados(data_choice)
    # 1.2 Mostrar informações básicas do dataset (número de amostras, entradas, saídas)
    dataset.show_info()

    # 2.1 Definir a arquitetura da MLP (entradas, ocultas, saidas)
    camadas = [dataset.num_entradas, num_neuronios_oculta, dataset.num_saidas]
    # 2.2 Criar a MLP com os pesos inicializados aleatoriamente
    mlp: MLP = MLP(camadas)

    # TODO: 3. Treinar a MLP (forward + backpropagation)
    mlp.treinar(dataset.treino, taxa_aprendizado, epocas)
    # TODO: 4. Testar a MLP e avaliar resultados


if __name__ == "__main__":
    main(
        data_choice=DataChoiceEnum.CARACTERES_COMPLETO,
        taxa_aprendizado=0.1,
        epocas=1000,
    )
