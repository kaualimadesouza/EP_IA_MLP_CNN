import random

from value_objects import Amostra


class Neuronio:
    def __init__(self, pesos: list[float], bias: float):
        self.pesos: list[float] = pesos
        self.bias: float = bias
        self.saida: float = 0.0

    def calcular_saida(self, entradas: list[float]):
        # Calcula a saída do neurônio
        # Formula: Somatorio de (peso * entrada) + bias
        soma_ponderada = 0

        # tuplas de (entrada, peso)
        entradas_pesos = zip(entradas, self.pesos)
        for entrada, peso in entradas_pesos:
            print(f"Entrada: {entrada}, Peso: {peso}")
            soma_ponderada += entrada * peso
            print("Soma ponderada atual:", soma_ponderada)
        self.saida = soma_ponderada + self.bias
        return self.saida


class Camada:
    def __init__(self, num_neuronios: int, num_entradas: int):
        self.neuronios = []
        for _ in range(num_neuronios):
            pesos = [random.uniform(-0.5, 0.5) for _ in range(num_entradas)]
            bias = 0.0
            self.neuronios.append(Neuronio(pesos, bias))


class MLP:
    def __init__(self, tamanhos_camadas: list[int]):
        self.tamanhos_camadas = tamanhos_camadas
        self.camadas: list[Camada] = []

        for i in range(1, len(tamanhos_camadas)):
            camada = Camada(tamanhos_camadas[i], tamanhos_camadas[i - 1])
            self.camadas.append(camada)

    def forward(self, entrada: list[float]) -> list[float]:
        # TODO: Propaga a entrada pela rede, camada por camada
        pass

    def backpropagation(self, esperado: list[float], taxa_aprendizado: float):
        # TODO: Calcula os deltas (erros) da camada de saida ate a oculta
        # TODO: Atualiza os pesos e bias de cada neuronio
        pass

    def treinar(
        self, dados_treino: list[Amostra], taxa_aprendizado: float, epocas: int
    ):
        # TODO: Loop de epocas
        # TODO: Para cada amostra: forward + backpropagation
        # TODO: Calcular e guardar o erro da epoca
        pass

    def testar(self, dados_teste: list[Amostra]):
        # TODO: Para cada amostra: forward e comparar com esperado
        pass

    def visualizar_arquitetura(self):
        print("=" * 50)
        print("          Arquitetura da MLP")
        print("=" * 50)
        print(f"  Camada de entrada: {self.tamanhos_camadas[0]} entradas")
        print("-" * 50)
        for i, camada in enumerate(self.camadas):
            tipo = "saida" if i == len(self.camadas) - 1 else "oculta"
            print(f"  Camada {i + 1} ({tipo}): {len(camada.neuronios)} neuronios")
            for j, neuronio in enumerate(camada.neuronios):
                pesos_fmt = [f"{p:.4f}" for p in neuronio.pesos]
                print(
                    f"    Neuronio {j + 1}: pesos={pesos_fmt}, bias={neuronio.bias:.4f}"
                )
            print("-" * 50)
        print("=" * 50)


if __name__ == "__main__":
    # Exemplo de uso
    MLP_teste = MLP([2, 3, 1, 7, 2])
    MLP_teste.visualizar_arquitetura()
