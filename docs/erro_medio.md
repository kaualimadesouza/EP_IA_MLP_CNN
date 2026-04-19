# Erro Médio — O Termômetro da Rede Neural

## A Ideia em Uma Frase

> O erro médio é um **único número** que resume "quanto a rede está errando" em todas as amostras. Você não usa ele pra treinar — você usa pra **saber se o treino está funcionando**.

---

## Por Que Precisamos Medir o Erro

Durante o treino, a rede passa por milhares de pequenos ajustes via backpropagation. Como saber se ela está realmente aprendendo ou se está travada, piorando, ou oscilando?

A resposta é acompanhar uma métrica global ao longo das épocas:

```
Epoca    1/1000 | erro: 0.521    ← ponto de partida
Epoca   50/1000 | erro: 0.283    ← caindo? bom!
Epoca  500/1000 | erro: 0.091
Epoca 1000/1000 | erro: 0.012    ← chegou perto de zero
```

Sem esse número, o treino vira uma caixa preta — você roda e reza.

---

## Como o Erro É Calculado

### Passo 1: erro por neurônio (erro quadrático)

Pra cada neurônio de saída, calcula a diferença entre esperado e real, e **eleva ao quadrado**:

```
erro_neuronio = (esperado - saida)²
```

Exemplo com uma amostra:

```
esperado: [1.0,  0.0,  0.0]
saida:    [0.7,  0.2,  0.3]

erros:
neuronio 0: (1.0 - 0.7)² = 0.09
neuronio 1: (0.0 - 0.2)² = 0.04
neuronio 2: (0.0 - 0.3)² = 0.09
```

**Por que elevar ao quadrado?**

1. **Remove o sinal**: errar pra cima e pra baixo contam como erro positivo.
2. **Penaliza erros grandes**: errar por `0.5` conta `0.25`; errar por `1.0` conta `1.0` (4× pior). Isso prioriza corrigir erros grandes.
3. **Permite cálculo de derivada suave**: essencial pro backprop.

### Passo 2: erro da amostra (soma dos neurônios)

Soma os erros de todos os neurônios daquela amostra:

```
erro_amostra = 0.09 + 0.04 + 0.09 = 0.22
```

### Passo 3: acumular sobre todas as amostras (erro da época)

Durante uma época, vai somando o erro de cada amostra:

```
erro_epoca = 0.0
for amostra in dados_treino:
    saida = forward(amostra.entrada)
    erro_epoca += sum((esp - s) ** 2 for esp, s in zip(amostra.esperado, saida))
```

Se a época tem 1060 amostras com erros variados, o `erro_epoca` é a soma de todos — tipo 45.7.

### Passo 4: dividir pelo número de amostras (erro médio)

O número bruto da soma depende do tamanho do dataset. Pra normalizar:

```
erro_medio = erro_epoca / len(dados_treino)
```

Exemplo: `45.7 / 1060 ≈ 0.043`. Agora você tem o **erro médio por amostra**, que é comparável entre datasets de tamanhos diferentes.

---

## O Nome Formal: MSE (Mean Squared Error)

O que você calcula é a métrica clássica de redes neurais:

```
MSE = (1/N) · Σ (esperado - saida)²
       │       │
       │       └── soma sobre todas amostras e todos neuronios
       └── divide pela quantidade total
```

Em português: **Erro Quadrático Médio**.

---

## Como Interpretar o Valor

A escala depende da faixa das saídas. Pra classificação com saídas em `(0, 1)` ou `(-1, +1)`:

| MSE                    | Interpretação                                       |
| ---------------------- | --------------------------------------------------- |
| **~0.5 ou mais**       | Rede praticamente chutando — não aprendeu nada      |
| **~0.25**              | Baseline de "saída média" sem aprendizado real      |
| **~0.05 a 0.10**       | Aprendendo, mas com erros frequentes                |
| **~0.01 a 0.05**       | Classificação razoável                              |
| **< 0.01**             | Excelente — quase memorizando o conjunto de treino  |

Pro EP, um bom objetivo é chegar a MSE `< 0.01` em datasets simples (AND, OR, XOR) e `< 0.05` em datasets complexos (caracteres).

---

## O Formato da Curva do Erro

A curva do erro ao longo das épocas revela muita coisa sobre o treino:

### Bom comportamento (aprendendo)

```
erro
 0.5 ┤●
     │ ●
     │  ●●
 0.3 ┤    ●●
     │       ●●
     │         ●●●
 0.1 ┤            ●●●●●
     │                 ●●●●●●●●
   0 ┤                         ●●●●●●●●●●
     └─────────────────────────────────────
     0        500       1000   época
```

Queda rápida no começo, depois desaceleração. É o padrão esperado — a rede acha os pesos "fáceis" primeiro e depois refina.

### Rede não está aprendendo (bug provável)

```
erro
 0.5 ┤●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
     │
     │
 0.1 ┤
   0 ┤
     └─────────────────────────────────────
     0        500       1000   época
```

Linha reta. Pesos não estão sendo atualizados. Possíveis causas:
- Backpropagation com bug (delta errado, update errado).
- Taxa de aprendizado zero ou muito próxima.
- Funções de ativação retornando sempre 0 (ex.: sigmoid saturada).

### Erro explodindo (instabilidade)

```
erro
10   ┤                            ●●●●●
     │                         ●●●
     │                      ●●●
 5   ┤                   ●●●
     │               ●●●
     │           ●●●
 0.5 ┤●●●●●●●●●
     └─────────────────────────────────────
     0        500       1000   época
```

Sinal de que a taxa de aprendizado está alta demais. A rede toma passos gigantes e passa do ponto ideal, voltando com inércia e piorando.

Solução: reduzir a `taxa_aprendizado` (ex.: de `0.5` pra `0.1`).

### Platô prematuro

```
erro
 0.5 ┤●
     │ ●
     │  ●●
 0.2 ┤   ●●●●●●●●●●●●●●●●●●●●●●●●
     │
   0 ┤
     └─────────────────────────────────────
     0        500       1000   época
```

Cai um pouco e estaciona num valor alto. Possíveis causas:
- Neurônios saturados (`a ≈ ±1` com tanh, derivada ~0).
- Arquitetura insuficiente (poucos neurônios ocultos pro problema).
- Taxa muito baixa.

---

## Erro Médio vs Delta — Duas Coisas Diferentes

Uma confusão comum: o erro médio **não é** o que o backpropagation usa pra atualizar pesos.

| Conceito         | Papel                                 | Usado por       |
| ---------------- | ------------------------------------- | --------------- |
| **erro_medio (MSE)** | Métrica global pra monitorar o treino | Você (humano)   |
| **δ (delta)**    | Gradiente local por neurônio          | Backpropagation |

O backprop olha apenas `(esperado - saida)` **dentro de cada amostra individual** pra calcular os δ. Ele nunca acumula nada na escala da época.

O erro médio é **só pra log**. Tirar ele não afeta o treino em nada — só te deixa cego.

---

## Variações Úteis

### Acurácia em vez de MSE (classificação)

Pra classificação one-hot, uma métrica mais interpretável:

```python
acertos = 0
for amostra in dados_treino:
    saida = self.forward(amostra.entrada)
    self.backpropagation(amostra.esperado, taxa)
    if saida.index(max(saida)) == amostra.esperado.index(max(amostra.esperado)):
        acertos += 1
acuracia = acertos / len(dados_treino)
```

Dá pra ler "86% de acerto" direto — não precisa interpretar uma escala abstrata.

### Separar erro de treino e erro de teste

Em projetos mais completos, você mede os dois:

```
Epoca 100 | erro_treino: 0.042 | erro_teste: 0.156
```

Se `erro_treino` cai mas `erro_teste` sobe, tem **overfitting** — a rede está memorizando o treino mas não generalizando.

Pro EP básico, medir só o erro de treino já é suficiente.

### Raiz do erro (RMSE)

Tirar a raiz quadrada do MSE dá o RMSE, que tem a mesma unidade da saída:

```python
rmse = math.sqrt(erro_medio)
```

Útil se você quer uma métrica "na mesma escala" dos valores. Pro EP, o MSE puro basta.

---

## Resumo em uma imagem mental

```
            ┌──────────────────────────────┐
            │        TREINAMENTO           │
            │                              │
ENTRADA ──→ │ forward → backprop → update  │ ──→ (rede um pouco melhor)
            │             ↑                │
            │             └─ usa δ local   │
            └──────────────────────────────┘
                      │
                      ▼ (entre uma amostra e outra,
                         acumula erro quadrático)
            ┌──────────────────────────────┐
            │         MONITORAMENTO        │
            │                              │
            │ erro_epoca += (esp - saida)² │
            │                              │
            │ no fim da epoca:             │
            │ erro_medio = erro_epoca / N  │
            │                              │
            │ → print pro humano ver       │
            └──────────────────────────────┘
```

O erro médio vive num "fluxo paralelo" ao treino. Ele observa mas não interfere.

---

## Por Que Fazer Isso Mesmo Sendo "Opcional"

1. **Debug**: quando a rede não aprende, a curva do erro é a primeira pista do problema.
2. **Comparação de hiperparâmetros**: "taxa 0.1 vs 0.5 — qual convergiu mais rápido?" só faz sentido com dados.
3. **Relatório acadêmico**: o professor provavelmente vai pedir um gráfico de convergência. Sem essa métrica, você não tem dado pra plotar.
4. **Condição de parada**: early stopping (`if erro_medio < alvo: break`) só funciona se você mede.

Custo pra implementar: ~3 linhas de código. Ganho: visibilidade total do processo.

---

## TL;DR

- **Erro médio** = soma dos erros quadráticos de todas as amostras, dividida pelo número de amostras.
- É o **termômetro** da rede: mostra se está aprendendo, parada, ou piorando.
- Não afeta o treino — o backprop usa `δ`, não usa o erro médio.
- Tirar ele deixa você cego; mantê-lo custa quase nada.
- Aprender a ler a curva dele é 80% do debug de rede neural.
