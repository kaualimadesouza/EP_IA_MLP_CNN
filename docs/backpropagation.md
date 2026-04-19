# Backpropagation — Como Funciona

## A Ideia em Uma Frase

> Backpropagation é o algoritmo que **distribui a culpa do erro** entre todos os pesos da rede, indo da saída pra entrada, e depois **ajusta cada peso** na direção certa pra errar menos.

---

## Analogia: a linha de montagem

Imagine uma fábrica de pão com 3 estações:

1. **Estação A** — mistura os ingredientes.
2. **Estação B** — amassa a massa.
3. **Estação C** — cozinha.

O pão sai queimado. Quem é o culpado?

- Pode ser o cozinheiro (tempo errado no forno) → ajuste a estação C.
- Pode ser que a massa chegou ruim → ajuste a estação B.
- Pode ser que os ingredientes estavam errados → ajuste a estação A.

O único jeito de descobrir é **voltar pelo caminho**: olhar o pão final, ver o que precisava ser diferente, e rastrear a culpa até o começo.

É exatamente isso que o backpropagation faz com uma rede neural.

---

## O Problema

Depois do forward, a rede produziu uma saída. Compare com o esperado:

```
esperado: [1.0,  0.0,  0.0]
saida:    [0.7,  0.3,  0.5]
erro:     [0.3, -0.3, -0.5]
```

Tem erro. Cada peso da rede contribuiu com esse erro de algum jeito — mas **quanto**?

Um peso lá da primeira camada oculta influenciou o resultado final passando por várias camadas. O backpropagation responde essa pergunta matematicamente usando a **regra da cadeia** das derivadas.

---

## As 3 Etapas

### Etapa 1: δ (delta) na camada de saída

Pros neurônios da **última camada**, você tem o erro diretamente: `esperado - saida`.

O **delta** de um neurônio = quanta culpa ele tem no erro.

```
δ = (esperado - saida) · saida · (1 - saida)
    └─────── erro ──────┘   └──── derivada da sigmoid ───┘
```

A parte `saida · (1 - saida)` é a **derivada da sigmoid**, que ajusta a culpa conforme a "sensibilidade" do neurônio naquele ponto.

### Etapa 2: propagar o δ pra trás

Neurônios das **camadas ocultas** não têm "esperado" direto. A culpa deles vem **emprestada** dos neurônios seguintes:

- Se o neurônio da próxima camada teve muita culpa (δ grande)
- E foi bastante influenciado por esse neurônio atual (peso grande)
- Então esse neurônio atual carrega uma fração dessa culpa

```
δ_atual = (Σ δ_proximos · pesos_proximos) · saida · (1 - saida)
```

Isso é feito de trás pra frente: penúltima camada → antepenúltima → ... → primeira.

### Etapa 3: ajustar pesos e biases

Agora que cada neurônio sabe sua culpa `δ`, atualize:

```
peso_novo = peso_antigo + taxa · δ · entrada_recebida
bias_novo = bias_antigo + taxa · δ
```

Onde `taxa` é a **taxa de aprendizado** (ex.: `0.1`), que controla o tamanho do passo.

**Intuição**: se um peso amplificou uma entrada que gerou erro, ele é reduzido; se amortizou uma entrada que deveria ter contribuído mais, ele é aumentado.

---

## Por Que Chamam de "Propagation"?

Porque o erro **flui de trás pra frente**, no sentido oposto do forward:

```
FORWARD  (calcula saida):
    entrada → camada 1 → camada 2 → camada 3 → saida

BACKWARD (calcula culpa):
    δ_1    ←   δ_2    ←   δ_3    ← (esperado - saida)
```

Cada camada só calcula seu `δ` **depois** que a camada seguinte já tiver o dela. Por isso começa pela última.

---

## O Papel do Forward

O forward não serve só pra calcular a saída — ele deixa uma **trilha** que o backprop usa:

| Salvo no neurônio          | Usado pelo backprop para…                    |
| -------------------------- | -------------------------------------------- |
| `self.saida` (a)           | Calcular a derivada da ativação `a·(1-a)`    |
| `self.entradas`            | Atualizar pesos (etapa 3)                    |
| `self.soma_ponderada` (z)  | Caso queira `f'(z)` explícito                |
| `self.pesos`               | Propagar `δ` pra camada anterior             |

Sem esses valores guardados, o backprop teria que recalcular tudo do zero — por isso o forward "com memória" é essencial.

---

## O Algoritmo Completo (Pseudocódigo)

```
treinar(dados_treino, taxa, epocas):
    para cada epoca:
        para cada amostra:
            forward(amostra.entrada)           # preenche saidas, entradas, z
            backpropagation(amostra.esperado, taxa)


backpropagation(esperado, taxa):
    # ETAPA 1: delta da ultima camada
    para cada neuronio da ultima camada:
        a = neuronio.saida
        neuronio.delta = (esperado[j] - a) · a · (1 - a)

    # ETAPA 2: propagar delta de tras pra frente
    para cada camada (da penultima ate a primeira):
        proxima_camada = camada seguinte
        para cada neuronio_j em camada:
            soma = 0
            para cada neuronio_k em proxima_camada:
                soma += neuronio_k.delta · neuronio_k.pesos[j]
            a = neuronio_j.saida
            neuronio_j.delta = soma · a · (1 - a)

    # ETAPA 3: atualizar pesos e biases
    para cada camada em self.camadas:
        para cada neuronio em camada:
            para cada i, peso em neuronio.pesos:
                neuronio.pesos[i] += taxa · neuronio.delta · neuronio.entradas[i]
            neuronio.bias += taxa · neuronio.delta
```

---

## Cuidados Comuns

1. **Ordem das etapas**: etapa 3 vem **depois** de todos os `δ` calculados. Se atualizar pesos no meio da etapa 2, a propagação vai usar pesos já alterados e dá errado.

2. **Índices dos pesos**: o peso `k→j` significa "peso do neurônio `k` (camada seguinte) que conecta ele à posição `j` (entrada `j` = saída do neurônio `j` da camada anterior)". No código: `neuronios[k].pesos[j]`.

3. **Sinal da equação**: aqui usamos `δ = (esperado - saida)·f'(z)` com update `peso += taxa·δ·entrada`. Se trocar pra `δ = (saida - esperado)·f'(z)`, use `peso -= taxa·δ·entrada`. Não misture.

4. **Usar a ativação certa**: `a·(1-a)` é derivada **da sigmoid**. Se um dia trocar pra `tanh`, é `1 - a²`; pra `ReLU`, é `1 se z>0 senão 0`.

---

## Por Que a Taxa de Aprendizado Importa

| Taxa       | Efeito                                                   |
| ---------- | -------------------------------------------------------- |
| Muito baixa (0.001) | Passos cautelosos. Aprende devagar, risco de empacar em mínimo local. |
| Alta (1.0)          | Passos grandes. Pode passar do ponto ideal e ficar oscilando. |
| Típica (0.01 – 0.5) | Equilíbrio razoável pra maioria dos problemas.          |

Comece com `0.1` e ajuste olhando a curva do erro.

---

## Sinais de Que Está Funcionando

Depois de implementar, rode o treino e olhe o erro médio por época:

**Aprendendo** (bom sinal):
```
Epoca   1/100 | erro: 0.521
Epoca  10/100 | erro: 0.283
Epoca  50/100 | erro: 0.091
Epoca 100/100 | erro: 0.012
```

**Parado** (bug provável):
```
Epoca   1/100 | erro: 0.521
Epoca  50/100 | erro: 0.521    ← pesos nao estao sendo atualizados
Epoca 100/100 | erro: 0.521
```

**Explodindo** (taxa alta demais ou sinal trocado):
```
Epoca   1/100 | erro: 0.521
Epoca  50/100 | erro: 2.341
Epoca 100/100 | erro: 8.920
```

---

## Estratégia de Teste

Na hora de validar sua implementação, teste em ordem crescente de dificuldade:

1. **AND / OR** — convergem em poucas épocas. Se esses não funcionam, o backprop tem bug sério.
2. **XOR** — exige não-linearidade; se funciona aqui, sua rede aprende funções complexas.
3. **Caracteres reduzido (Fausett)** — próximo degrau, 7 classes.
4. **Caracteres completo** — 26 classes, validação final.

Pular etapas só atrasa debug — XOR com MSE baixo é o melhor "atestado" de que seu backprop está correto.

---

## TL;DR

Backpropagation é só **regra da cadeia aplicada várias vezes**, organizada em 3 etapas:

1. **Onde errou?** → compara saída com esperado na última camada.
2. **Quem é culpado?** → distribui a culpa pra trás, via os pesos.
3. **Como corrigir?** → ajusta cada peso proporcional à culpa do seu neurônio e à entrada que ele recebeu.

A "mágica" está em usar a **memória do forward** pra não recalcular nada — só propagar o erro no caminho inverso.
