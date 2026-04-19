# Erro Quadrático — Explicação do Zero

> Este doc explica **o que é um "erro quadrático"**, partindo da estaca zero. Nenhum pré-requisito além de saber o que é uma subtração e o que é elevar ao quadrado.

---

## Começando pelo começo: o que é "erro"?

Imagine que você previu que ia chover 10 mm hoje, mas choveu 7 mm. Seu erro foi:

```
erro = esperado - obtido = 10 - 7 = 3 mm
```

Simples assim. **Erro é a diferença entre o que você esperava e o que realmente aconteceu**.

Numa rede neural, a mesma ideia:

```
esperado = [1, 0, 0]     <- o que deveria sair
saida    = [0.7, 0.2, 0.3]  <- o que realmente saiu

erros:
  posicao 0: 1 - 0.7 = +0.3    <- errou pra baixo por 0.3
  posicao 1: 0 - 0.2 = -0.2    <- errou pra cima por 0.2
  posicao 2: 0 - 0.3 = -0.3    <- errou pra cima por 0.3
```

Até aí, tudo bem. Mas agora surge um problema.

---

## O problema: erros com sinais diferentes se cancelam

Se você quer **somar** os erros pra ter "o erro total da amostra", olha o que acontece:

```
erro total = +0.3 + (-0.2) + (-0.3) = -0.2
```

Os erros positivos e negativos **se cancelaram parcialmente**. O resultado (-0.2) sugere que o erro foi pequeno, mas **não foi pequeno**: a rede errou 0.3 + 0.2 + 0.3 = 0.8 em magnitude total!

**Isso é um problema sério.** Uma rede que erra muito mas equilibrado (metade pra cima, metade pra baixo) ia parecer que erra pouco. Precisamos de uma forma de tornar **todo erro positivo** antes de somar.

---

## A solução: elevar ao quadrado

Elevar um número ao quadrado (multiplicá-lo por ele mesmo) tem uma propriedade mágica: **sempre dá positivo**.

```
(+0.3)² = 0.3 · 0.3 = 0.09
(-0.3)² = (-0.3) · (-0.3) = 0.09     <- negativo vezes negativo = positivo!
(+0.5)² = 0.25
(-0.5)² = 0.25
```

O sinal **some**. Só resta a magnitude (o tamanho) do erro.

Então, o **erro quadrático** é simplesmente:

```
erro_quadratico = (esperado - obtido)²
```

Lê-se "erro ao quadrado".

---

## Exemplo numérico completo

Voltando aos nossos dados:

```
esperado = [1,   0,   0  ]
saida    = [0.7, 0.2, 0.3]
```

### Sem quadrado (só diferença)

```
pos 0: 1 - 0.7 = +0.3
pos 1: 0 - 0.2 = -0.2
pos 2: 0 - 0.3 = -0.3

soma = +0.3 - 0.2 - 0.3 = -0.2     ← parece pequeno, mas e' enganoso
```

### Com quadrado (erro quadrático)

```
pos 0: (1 - 0.7)² = (+0.3)² = 0.09
pos 1: (0 - 0.2)² = (-0.2)² = 0.04
pos 2: (0 - 0.3)² = (-0.3)² = 0.09

soma = 0.09 + 0.04 + 0.09 = 0.22    ← reflete o erro real!
```

Com o quadrado, todo erro contribui positivamente. Não tem como um neurônio "cancelar" o erro do outro.

---

## "Mas por que o quadrado? Não bastava o módulo?"

Boa pergunta. Uma alternativa pra tirar o sinal é usar **valor absoluto** (módulo):

```
|+0.3| = 0.3
|-0.3| = 0.3
```

Ambos funcionam pra "eliminar o sinal". A diferença está em **como eles tratam erros grandes vs pequenos**.

### Comparação: quadrado vs módulo

| Erro real | Módulo `|erro|` | Quadrado `erro²` |
| --------- | --------------- | ---------------- |
| 0.1       | 0.1             | 0.01             |
| 0.5       | 0.5             | 0.25             |
| 1.0       | 1.0             | 1.00             |
| 2.0       | 2.0             | 4.00             |
| 5.0       | 5.0             | 25.0             |

Olha o que acontece quando o erro dobra (de 0.5 pra 1.0):

- **Módulo**: o erro duplica (0.5 → 1.0).
- **Quadrado**: o erro **quadruplica** (0.25 → 1.0).

E quando o erro fica cinco vezes maior (0.1 → 0.5):

- **Módulo**: fica 5× maior (0.1 → 0.5).
- **Quadrado**: fica **25×** maior (0.01 → 0.25).

### O que isso significa

O quadrado **penaliza erros grandes muito mais** que erros pequenos. Isso é exatamente o que queremos numa rede neural:

> *"Prefiro errar um pouquinho em várias amostras do que errar muito em uma única amostra."*

Se você usa módulo, errar 1 por muito ou errar 2 por pouco dá o mesmo erro total (4). Se você usa quadrado, errar uma amostra por 2 (= 4 de custo) é muito pior que errar 4 amostras por 1 (= 4 de custo). Isso incentiva a rede a **nunca deixar nenhum erro crescer muito**, espalhando o erro em vez de concentrá-lo.

### A outra razão (mais técnica)

O quadrado é uma função **suave e diferenciável** em todo lugar, inclusive no zero. O módulo **não é diferenciável no zero** (tem um "bico"). Como o backpropagation precisa da derivada do erro, o quadrado é muito mais conveniente matematicamente.

Em resumo:

| Motivo                        | Quadrado | Módulo  |
| ----------------------------- | -------- | ------- |
| Remove sinal                  | Sim      | Sim     |
| Penaliza erros grandes mais   | **Sim**  | Não     |
| Derivável no zero             | **Sim**  | Não     |

O quadrado ganha em duas frentes, por isso virou padrão.

---

## Visualizando graficamente

Imagine o eixo X como "erro" (positivo ou negativo) e Y como "custo":

```
           |
   8 ──────┤
           │                        ●
   6 ──────┤                     ●
           │                   ●
   4 ──────┤                 ●   ← quadrado: sobe rapido
           │                ●
   2 ──────┤              ●
           │            ●
   0 ──────┼──●●●●●●●●●────────────
           │  -3  -2  -1  0  1  2  3   ← erro
```

A curva `y = x²` tem formato de "tigela": é zero exatamente quando o erro é zero, e **sobe cada vez mais rápido** conforme o erro aumenta. Pra erros pequenos (x perto de zero), a tigela é quase plana; pra erros grandes, ela é bem íngreme.

É por isso que chamamos de **função convexa**: ela tem um único ponto mínimo (o zero) e tudo converge pra ele. Muito conveniente pra otimização.

---

## Juntando tudo: erro quadrático em um neurônio de rede neural

Em cada neurônio da camada de saída:

```
erro = esperado - saida
erro_quadratico = erro ²  =  (esperado - saida)²
```

Em código Python:

```python
erro = esperado[j] - neuronio.saida
erro_quadratico = erro ** 2
```

Ou numa linha só:

```python
erro_quadratico = (esperado[j] - neuronio.saida) ** 2
```

É literalmente uma subtração e uma multiplicação por si mesmo. Nada de matemática avançada.

---

## Do erro quadrático pro MSE

Agora que você entende o erro quadrático, o MSE (Mean Squared Error) é só **a média dele**:

```
MSE = (1/N) · Σ (esperado - saida)²
```

Tradução:

- **Σ (esperado - saida)²** → soma dos erros quadráticos de todas as amostras e todos os neurônios.
- **1/N** → divide pelo número de amostras pra ter uma média.

Passo a passo:

```
1. Pra cada amostra, calcula o erro quadratico de cada neuronio.
2. Soma tudo dentro de uma amostra (erro total da amostra).
3. Soma todas as amostras (erro total da epoca).
4. Divide pelo numero de amostras (media).
```

Resultado: um **único número** que representa "quanto a rede errou em média nessa época".

---

## Resumindo em linguagem simples

- **Erro** = diferença entre esperado e obtido (pode ser positivo ou negativo).
- **Erro quadrático** = erro elevado ao quadrado (sempre positivo, penaliza erros grandes mais).
- **MSE** = média dos erros quadráticos (um número só, fácil de monitorar).

A próxima vez que você ler uma fórmula com `(esperado - saida)²`, é literalmente isso: uma subtração multiplicada por ela mesma. Nada místico.

---

## Perguntas que talvez você tenha agora

**"Por que dá pra elevar valores decimais ao quadrado? Quadrado não é pra número inteiro?"**

Dá sim. `0.3² = 0.3 × 0.3 = 0.09`. Funciona pra qualquer número real — inteiro, decimal, negativo. Em Python: `0.3 ** 2` retorna `0.09`.

**"Se o erro é 0.1, elevar ao quadrado dá 0.01 — ficou menor! Não deveria ficar maior?"**

Boa observação. Pra números entre -1 e +1, elevar ao quadrado **diminui** o valor absoluto. Pra números fora desse intervalo, o quadrado aumenta. Isso é propriedade da função `x²`. Não é um problema — o que importa pro treino é a **comparação relativa** entre erros, e essa comparação continua fazendo sentido (um erro maior continua tendo quadrado maior).

**"O MSE pode ser negativo?"**

Não. Como é média de valores ao quadrado (sempre ≥ 0), o MSE é sempre ≥ 0. MSE = 0 significa que a rede acertou perfeitamente em todas as amostras. MSE grande significa que a rede está errando feio.

**"Por que o nome 'quadrático' e não 'ao quadrado'?"**

"Quadrático" é o adjetivo que vem de "quadrado". Uma expressão que envolve `x²` é chamada de expressão quadrática (mesma raiz etimológica de "quadrado" — forma geométrica de 4 lados iguais). É só convenção da matemática.

---

## TL;DR de 3 linhas

1. **Erro** é a diferença entre esperado e obtido.
2. **Elevar ao quadrado** torna tudo positivo E penaliza mais erros grandes.
3. **MSE** é a média dos erros quadráticos — um número só pra monitorar a rede.
