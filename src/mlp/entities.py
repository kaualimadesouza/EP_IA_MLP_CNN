import math
import random

from value_objects import Amostra, ResultadoTeste


def tanh(x: float) -> float:
    """Função de ativação tangente hiperbolica. Saida em (-1, +1)."""
    return math.tanh(x)


def tanh_derivada_de_saida(a: float) -> float:
    """Derivada da tanh a partir da saida ja calculada: tanh'(z) = 1 - tanh(z)^2 = 1 - a^2."""
    return 1 - a**2


class Neuronio:
    def __init__(self, pesos: list[float], bias: float):
        self.pesos: list[float] = pesos
        self.bias: float = bias
        self.entradas: list[float] = []
        self.soma_ponderada: float = 0.0
        self.saida: float = 0.0
        # usado no backpropagation
        self.delta: float = 0.0

    def calcular_saida(self, entradas: list[float]) -> float:
        self.entradas = entradas

        # Variavel local pra nao acumular entre chamadas (erro classico com self.*).
        soma = 0.0
        for entrada, peso in zip(entradas, self.pesos):
            soma += entrada * peso

        # z = Σ(w·x) + b  -> pre-ativacao (usada no backprop pra f'(z))
        self.soma_ponderada = soma + self.bias
        # a = f(z)  -> saida do neuronio, aperta z em (-1, +1).
        self.saida = tanh(self.soma_ponderada)
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
        ativacao = entrada
        for camada in self.camadas:
            novas_saidas = []
            for neuronio in camada.neuronios:
                neuronio.calcular_saida(ativacao)
                novas_saidas.append(neuronio.saida)
            ativacao = novas_saidas
        return ativacao

    def backpropagation(self, esperado: list[float], taxa_aprendizado: float):
        # 1. Calcular delta da camada de saída
        for i, neuronio in enumerate(self.camadas[-1].neuronios):
            erro = esperado[i] - neuronio.saida
            neuronio.delta = erro * tanh_derivada_de_saida(neuronio.saida)

        # 2. Calcular delta das camadas ocultas (de trás pra frente)
        # 2.1 Iteramos de tras para frente, pulando a camada de saída (já calculada).
        for j in range(len(self.camadas) - 2, -1, -1):
            camada_atual = self.camadas[j]
            camada_acima = self.camadas[j + 1]

            for i, neuronio in enumerate(camada_atual.neuronios):
                # O delta do neuronio atual depende dos deltas da camada acima e dos pesos.
                erro = 0.0
                for neuronio_acima in camada_acima.neuronios:
                    erro += neuronio_acima.delta * neuronio_acima.pesos[i]
                neuronio.delta = erro * tanh_derivada_de_saida(neuronio.saida)

        # 3. Atualizar pesos e bias de todas as camadas
        for camada in self.camadas:
            for neuronio in camada.neuronios:
                for i, entrada in enumerate(neuronio.entradas):
                    # Atualização do peso: w = w (peso anterior) + taxa_aprendizado * delta * entrada
                    neuronio.pesos[i] = (
                        neuronio.pesos[i] + taxa_aprendizado * neuronio.delta * entrada
                    )
                # Atualização do bias: b = b (bias anterior) + taxa_aprendizado * delta
                neuronio.bias = neuronio.bias + taxa_aprendizado * neuronio.delta

    def treinar(
        self,
        dados_treino: list[Amostra],
        taxa_aprendizado: float,
        epocas: int,
    ) -> list[float]:
        """Treina a rede e retorna o historico de MSE por epoca."""
        historico_erro: list[float] = []
        # Padding pro numero da epoca ficar alinhado (ex: "   1", "  42", "1000").
        largura = len(str(epocas))
        for epoca in range(epocas):
            erro_epoca = 0.0
            for amostra in dados_treino:
                saida = self.forward(amostra.entrada)
                self.backpropagation(amostra.esperado, taxa_aprendizado)

                for s, esp in zip(saida, amostra.esperado):
                    erro_epoca += (esp - s) ** 2
            erro_medio = erro_epoca / len(dados_treino)
            historico_erro.append(erro_medio)
            print(f"Epoca {epoca + 1:>{largura}}/{epocas} - MSE: {erro_medio:.6f}")
        return historico_erro

    def testar(self, dados_teste: list[Amostra]) -> list[ResultadoTeste]:
        """Roda forward em cada amostra de teste e retorna lista de ResultadoTeste."""
        resultados: list[ResultadoTeste] = []
        for amostra in dados_teste:
            saida = self.forward(amostra.entrada)
            if len(amostra.esperado) == 1:
                # Classificacao binaria bipolar: sinal da saida decide
                predita = 1 if saida[0] > 0 else -1
                esperada = 1 if amostra.esperado[0] > 0 else -1
            else:
                # Multiclasse: indice do maior valor
                predita = saida.index(max(saida))
                esperada = amostra.esperado.index(max(amostra.esperado))
            resultados.append(
                ResultadoTeste(
                    entrada=amostra.entrada,
                    esperado_raw=amostra.esperado,
                    saida_raw=saida,
                    classe_predita=predita,
                    classe_esperada=esperada,
                    acerto=(predita == esperada),
                )
            )
        return resultados

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
