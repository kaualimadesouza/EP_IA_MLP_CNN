# Implementando o MSE (Erro Médio Quadrático)

> Este documento foca em **como implementar** o MSE no seu código, atendendo o requisito da professora: *"Um arquivo contendo o erro cometido pela rede neural em cada iteração do treinamento"*.
> Pra entender o **conceito** de MSE, veja `erro_medio.md`.

---

## Visão Geral

Você vai precisar:

1. **Calcular** o MSE a cada época durante o treino.
2. **Guardar** todos os MSEs num array ao longo das épocas.
3. **Salvar** esse array num arquivo CSV no fim do treino.
4. (Opcional) **Mostrar** o MSE no console durante o treino, pra acompanhar a convergência.

---

## Passo 1 — Calcular o MSE em uma época

Dentro do loop de `treinar`, você acumula o erro quadrático de cada amostra e divide pelo total.

```python
erro_epoca = 0.0
for amostra in dados_treino:
    saida = self.forward(amostra.entrada)
    self.backpropagation(amostra.esperado, taxa_aprendizado)

    # Soma dos erros quadraticos dos N neuronios de saida desta amostra
    erro_amostra = sum(
        (esp - s) ** 2
        for esp, s in zip(amostra.esperado, saida)
    )
    erro_epoca += erro_amostra

# MSE = media por amostra
mse = erro_epoca / len(dados_treino)
```

### Por que essa fórmula?

```
MSE = (1 / N) · Σ (esperado − saida)²
       │          │
       │          └── soma sobre todas as amostras e todos os neuronios de saida
       └── divide pelo numero de amostras (N)
```

- **Elevar ao quadrado**: remove o sinal e penaliza erros grandes.
- **Somar os neurônios**: junta o erro de todas as saídas da mesma amostra.
- **Somar as amostras**: acumula pela época inteira.
- **Dividir por N**: normaliza pra ficar comparável entre datasets de tamanhos diferentes.

---

## Passo 2 — Guardar o histórico num array

Crie uma lista que guarda o MSE de cada época:

```python
def treinar(self, dados_treino, taxa_aprendizado, epocas):
    historico_erro: list[float] = []      # <-- novo

    for epoca in range(epocas):
        erro_epoca = 0.0
        for amostra in dados_treino:
            saida = self.forward(amostra.entrada)
            self.backpropagation(amostra.esperado, taxa_aprendizado)
            erro_epoca += sum(
                (esp - s) ** 2 for esp, s in zip(amostra.esperado, saida)
            )
        mse = erro_epoca / len(dados_treino)
        historico_erro.append(mse)         # <-- guarda cada MSE

    return historico_erro                   # <-- devolve pra quem chamou
```

O `return` faz com que o `main.py` possa pegar o histórico e salvar em arquivo.

---

## Passo 3 — Salvar num CSV

A professora pede um arquivo com o erro por iteração. CSV é o formato mais simples e universal.

```python
import csv
from pathlib import Path

def salvar_erro_csv(historico: list[float], caminho: str) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoca", "mse"])
        for i, mse in enumerate(historico, start=1):
            writer.writerow([i, f"{mse:.6f}"])
```

Exemplo de uso:

```python
historico = mlp.treinar(dataset.treino, taxa_aprendizado, epocas)
salvar_erro_csv(historico, "saidas/erro_por_epoca.csv")
```

### Como o arquivo fica

```
epoca,mse
1,1.234567
2,0.987654
3,0.823411
...
999,0.011234
1000,0.010987
```

Cada linha = uma época. Fácil de abrir no Excel, LibreOffice, ou importar pra Python/R pra plotar.

---

## Passo 4 — Mostrar no console (opcional, mas útil)

Pra acompanhar durante o treino sem spammar o terminal, imprime a cada N épocas:

```python
for epoca in range(epocas):
    # ... calcula mse como antes ...

    historico_erro.append(mse)

    # Imprime a cada 50 epocas e sempre na primeira/ultima
    if (epoca + 1) % 50 == 0 or epoca == 0 or epoca == epocas - 1:
        print(f"Epoca {epoca + 1:>4}/{epocas} | MSE: {mse:.6f}")
```

Saída típica:

```
Epoca    1/1000 | MSE: 1.234567
Epoca   50/1000 | MSE: 0.234510
Epoca  100/1000 | MSE: 0.089123
...
Epoca 1000/1000 | MSE: 0.011234
```

Com isso, você enxerga a rede convergindo em tempo real.

---

## Implementação Completa

Juntando tudo, a função `treinar` fica assim:

```python
def treinar(
    self,
    dados_treino: list[Amostra],
    taxa_aprendizado: float,
    epocas: int,
) -> list[float]:
    """Treina a MLP e retorna o historico de MSE por epoca."""
    historico_erro: list[float] = []

    for epoca in range(epocas):
        erro_epoca = 0.0
        for amostra in dados_treino:
            saida = self.forward(amostra.entrada)
            self.backpropagation(amostra.esperado, taxa_aprendizado)
            erro_epoca += sum(
                (esp - s) ** 2 for esp, s in zip(amostra.esperado, saida)
            )
        mse = erro_epoca / len(dados_treino)
        historico_erro.append(mse)

        if (epoca + 1) % 50 == 0 or epoca == 0:
            print(f"Epoca {epoca + 1:>4}/{epocas} | MSE: {mse:.6f}")

    return historico_erro
```

E no `main.py`:

```python
import csv
from pathlib import Path

def salvar_erro_csv(historico, caminho):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoca", "mse"])
        for i, mse in enumerate(historico, start=1):
            writer.writerow([i, f"{mse:.6f}"])


def main(...):
    # ... carregar dados, criar MLP ...
    historico = mlp.treinar(dataset.treino, taxa_aprendizado, epocas)
    salvar_erro_csv(historico, "saidas/erro_por_epoca.csv")
```

---

## Lendo o Arquivo Depois

### Pra olhar rápido no terminal

```bash
head -20 saidas/erro_por_epoca.csv    # primeiras 20 epocas
tail -20 saidas/erro_por_epoca.csv    # ultimas 20 epocas
```

### Pra plotar em Python (se quiser)

```python
import csv
import matplotlib.pyplot as plt

epocas, mses = [], []
with open("saidas/erro_por_epoca.csv") as f:
    next(f)                            # pula cabecalho
    for row in csv.reader(f):
        epocas.append(int(row[0]))
        mses.append(float(row[1]))

plt.plot(epocas, mses)
plt.xlabel("Epoca")
plt.ylabel("MSE")
plt.title("Convergencia do treino")
plt.yscale("log")                       # escala log costuma facilitar a leitura
plt.savefig("saidas/curva_erro.png")
```

> `matplotlib` é opcional — usa se quiser um gráfico pra colocar no relatório/vídeo. A professora aceita "bibliotecas de PLOT" pela especificação.

---

## Variantes Possíveis

### Salvar direto durante o treino (menos memória)

Se você não quiser acumular tudo numa lista, escreva no arquivo a cada época:

```python
def treinar(self, dados_treino, taxa_aprendizado, epocas, caminho_erro):
    with open(caminho_erro, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoca", "mse"])

        for epoca in range(epocas):
            erro_epoca = 0.0
            for amostra in dados_treino:
                saida = self.forward(amostra.entrada)
                self.backpropagation(amostra.esperado, taxa_aprendizado)
                erro_epoca += sum(
                    (esp - s) ** 2 for esp, s in zip(amostra.esperado, saida)
                )
            mse = erro_epoca / len(dados_treino)
            writer.writerow([epoca + 1, f"{mse:.6f}"])
```

Vantagem: se o programa travar no meio, você já tem os dados parciais salvos.

### RMSE em vez de MSE

Raiz do MSE, fica na mesma unidade da saída:

```python
import math
rmse = math.sqrt(mse)
```

Útil pra relatório ("erro médio ~0.1 por saída"), mas pra a professora o MSE puro já atende.

### Calcular MSE no teste também

Além de medir no treino, mede no conjunto de teste **depois** do treino. Isso ajuda a detectar overfitting:

```python
def testar(self, dados_teste: list[Amostra]) -> float:
    erro_total = 0.0
    for amostra in dados_teste:
        saida = self.forward(amostra.entrada)
        erro_total += sum(
            (esp - s) ** 2 for esp, s in zip(amostra.esperado, saida)
        )
    return erro_total / len(dados_teste)
```

Se `MSE_treino << MSE_teste`, a rede decorou mas não aprendeu.

---

## Checklist de Implementação

- [ ] `treinar` calcula MSE a cada época.
- [ ] `treinar` acumula os MSEs num `historico_erro`.
- [ ] `treinar` retorna o histórico.
- [ ] `main` chama `salvar_erro_csv(historico, "saidas/erro_por_epoca.csv")`.
- [ ] Arquivo CSV tem cabeçalho `epoca,mse`.
- [ ] Console mostra o MSE a cada 50 épocas (ou escolha similar).
- [ ] (Opcional) Gráfico PNG plotando a curva.

---

## Armadilhas Comuns

**1. Esquecer de zerar `erro_epoca` a cada época.**

```python
erro_epoca = 0.0          # ← precisa estar DENTRO do loop de epocas
for epoca in range(epocas):
    for amostra in ...
```

Se zerar fora, você acumula o erro de todas as épocas juntas. Resultado vai crescendo monotonicamente, sem valor diagnóstico.

**2. Usar a saída antes do backprop em vez de depois.**

O MSE reflete o estado da rede **durante** a época. Ambos funcionam (usar `saida` do forward atual ou refazer forward depois do backprop), mas seja consistente: se mudar a convenção no meio do treino, a curva fica descontínua.

A convenção mais comum: **medir o erro antes de atualizar os pesos** (o `saida` do forward que você já tem, sem refazer).

**3. Não dividir pelo número de amostras.**

Sem dividir, o "erro" cresce com o tamanho do dataset. Trocar de `portas_logicas` (4 amostras) pra `caracteres_completo` (1060 amostras) mudaria a escala do número — inútil pra comparação.

**4. MSE baixo mas rede classifica mal.**

Não é bug — MSE mede distância contínua, não acerto discreto. Pra classificação, complementar com **acurácia** (matriz de confusão no teste) dá o panorama completo.

---

## TL;DR

- MSE = soma de `(esperado - saida)²` dividida pelo número de amostras.
- Calcula a cada época, guarda num array, salva num CSV no fim.
- Mostra no console a cada 50 épocas pra acompanhar convergência.
- Arquivo `saidas/erro_por_epoca.csv` com colunas `epoca, mse` atende o requisito da professora.
- A curva deve cair ao longo das épocas — se não cair, tem bug no treino.
