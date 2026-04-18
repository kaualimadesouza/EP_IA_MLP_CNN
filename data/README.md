# Datasets

Três famílias de dados usadas pelo EP:

1. **`portas_logicas/`** — tabelas verdade de AND/OR/XOR (sanity check da MLP).
2. **`caracteres_reduzido/`** e **`caracteres_reduzido_alt/`** — dataset clássico de Fausett (7 letras em grade 9×7).
3. **`caracteres_completo/`** — alfabeto completo A–Z em grade 10×12 (1.326 amostras).

Todos os valores de entrada estão em **representação bipolar** (`-1` e `+1`).

---

## `portas_logicas/`

Três portas lógicas, uma por arquivo. Cada linha é uma entrada da tabela verdade.

| Arquivo                | Linhas | Colunas | Descrição                                                |
| ---------------------- | ------ | ------- | -------------------------------------------------------- |
| `and.csv`              | 4      | 3       | Porta AND. Colunas: `x1, x2, y`. Só é `+1` quando ambos entram `+1`. |
| `or.csv`               | 4      | 3       | Porta OR. `+1` quando ao menos uma entrada é `+1`.       |
| `xor.csv`              | 4      | 3       | Porta XOR. `+1` quando as entradas diferem (problema não linearmente separável — exige camada oculta). |
| `portas logicas.zip`   | —      | —       | Pacote original enviado pelo professor.                  |

**Estrutura**: `entrada_1, entrada_2, saida_esperada` — `num_entradas = 2`, `num_saidas = 1`.

---

## `caracteres_reduzido/` (Fausett)

Dataset clássico do livro *Fausett — Fundamentals of Neural Networks*: **7 letras** (A, B, C, D, E, J, K) renderizadas numa grade **9×7 = 63 pixels**, com saída one-hot de 7 classes. Cada CSV tem **21 linhas** (3 variações × 7 letras) e **70 colunas** (63 entradas + 7 saídas).

| Arquivo                 | Linhas | Colunas | Descrição                                                      |
| ----------------------- | ------ | ------- | -------------------------------------------------------------- |
| `limpo.csv`             | 21     | 70      | Padrões **sem ruído**. Usado como conjunto de **treino**.      |
| `ruido.csv`             | 21     | 70      | Padrões com **ruído leve** (pixels invertidos). Usado em teste. |
| `ruido20.csv`           | 21     | 70      | Padrões com **~20% de ruído**. Usado em teste.                 |
| `caracteres-Fausett.zip`| —      | —       | Pacote original.                                               |

**Estrutura por linha**: 63 valores de pixel (−1/+1) + 7 valores one-hot (`+1` na classe correta, `−1` nas outras).

No `datasets.py`, o split é:
- **Treino**: `limpo.csv` (21 amostras)
- **Teste**: `ruido.csv` + `ruido20.csv` (42 amostras)

---

## `caracteres_reduzido_alt/`

Mesma estrutura do `caracteres_reduzido/` (7 letras, 9×7, 70 colunas), mas com **padrões diferentes** — versão alternativa fornecida pelo professor para comparar generalização.

| Arquivo                                 | Linhas | Colunas | Descrição                       |
| --------------------------------------- | ------ | ------- | ------------------------------- |
| `limpo.csv`                             | 21     | 70      | Padrões limpos (alternativos).  |
| `ruido.csv`                             | 21     | 70      | Padrões com ruído leve.         |
| `ruido20.csv`                           | 21     | 70      | Padrões com ~20% de ruído.      |
| `caracteres-fausett alternativo.zip`    | —      | —       | Pacote original.                |

> Atualmente **não está ligado** ao `DataChoiceEnum` — se quiser usar, basta adicionar uma entrada nova (ex.: `CARACTERES_REDUZIDO_ALT`) e um ramo no `PATHS_DATASETS`.

---

## `caracteres_completo/`

Alfabeto completo **A–Z** em grade **10×12 = 120 pixels**, com **51 amostras por letra** → **1.326 amostras** no total. Saída é one-hot de 26 classes.

| Arquivo                    | Formato     | Shape              | Descrição                                                                 |
| -------------------------- | ----------- | ------------------ | ------------------------------------------------------------------------- |
| `X.txt`                    | CSV (texto) | 1.326 × 120        | Matriz de entradas. Tem vírgula trailing → pandas lê 121 colunas e o `_carregar_caracteres_completo` remove a coluna NaN com `dropna(axis=1, how="all")`. |
| `Y_letra.txt`              | Texto       | 1.325 linhas       | Letra correspondente de cada amostra (`A`, `B`, …, `Z`), uma por linha. O código converte pra one-hot com `pd.get_dummies`. |
| `X.npy`                    | NumPy       | 1.326 × 120        | Mesma matriz de `X.txt`, já em formato binário (`np.load` direto).        |
| `Y_classe.npy`             | NumPy       | 1.326 × 26         | Rótulos já em formato one-hot (NumPy).                                    |
| `X_png.zip`                | ZIP         | —                  | Imagens PNG de cada amostra — útil pra visualizar o que a rede está vendo. |
| `CARACTERES COMPLETO.zip`  | ZIP         | —                  | Pacote original enviado pelo professor.                                   |

**Split no `datasets.py`**: shuffle com seed `42` + corte 80/20 → ~1.060 treino / ~266 teste.

> ⚠️ `X.txt` tem uma linha a mais do que `Y_letra.txt`. O `pd.concat` alinha por índice, então a amostra extra fica com label NaN. Para ficar 100% robusto, vale um `.dropna()` após o concat ou um `assert len(X) == len(Y)`.

---

## Resumo rápido

| Dataset                   | Entradas | Saídas | Treino | Teste |
| ------------------------- | -------- | ------ | ------ | ----- |
| `portas_logicas` (AND/OR/XOR) | 2        | 1      | 4      | 4     |
| `caracteres_reduzido`     | 63       | 7      | 21     | 42    |
| `caracteres_reduzido_alt` | 63       | 7      | 21     | 42    |
| `caracteres_completo`     | 120      | 26     | ~1.060 | ~266  |
