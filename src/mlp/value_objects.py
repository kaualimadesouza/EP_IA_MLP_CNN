from dataclasses import dataclass
from enum import Enum


@dataclass
class Amostra:
    entrada: list[float]
    esperado: list[float]


@dataclass
class Dataset:
    treino: list[Amostra]
    teste: list[Amostra]
    num_entradas: int
    num_saidas: int

    def show_info(self) -> None:
        """Imprime um resumo estilizado do dataset no terminal."""
        print(_formatar_dataset(self))


class DataChoiceEnum(Enum):
    OR = "OR"
    AND = "AND"
    XOR = "XOR"
    CARACTERES_REDUZIDO = "CARACTERES_REDUZIDO"
    CARACTERES_COMPLETO = "CARACTERES_COMPLETO"


# ---------------------------------------------------------------------------
# Helpers de exibição (uso interno de Dataset.show_info)
# ---------------------------------------------------------------------------
_LARGURA = 64
_CIANO = "\033[96m"
_AMARELO = "\033[93m"
_VERDE = "\033[92m"
_MAGENTA = "\033[95m"
_CINZA = "\033[90m"
_NEGRITO = "\033[1m"
_RESET = "\033[0m"


@dataclass
class _Metrica:
    rotulo: str
    valor: int


@dataclass
class _Linha:
    """Linha da moldura: `plain` calcula padding; `colorida` e' exibida."""

    plain: str
    colorida: str

    @classmethod
    def destaque(cls, texto: str) -> "_Linha":
        return cls(plain=texto, colorida=f"{_NEGRITO}{texto}{_RESET}")

    @classmethod
    def de_metrica(cls, m: _Metrica, largura_rot: int) -> "_Linha":
        rot = m.rotulo.ljust(largura_rot)
        return cls(
            plain=f"{rot} : {m.valor}",
            colorida=f"{_AMARELO}{rot}{_RESET} {_CINZA}:{_RESET} {_VERDE}{m.valor}{_RESET}",
        )

    @classmethod
    def de_amostra(cls, nome: str, valores: list[float]) -> "_Linha":
        n = len(valores)
        rot = f"  {nome.ljust(8)} ({n}): "
        preview = _preview(valores, _LARGURA - 4 - len(rot))
        return cls(
            plain=f"{rot}{preview}",
            colorida=(
                f"  {_AMARELO}{nome.ljust(8)}{_RESET} "
                f"({_CINZA}{n}{_RESET}): {_MAGENTA}{preview}{_RESET}"
            ),
        )

    def na_moldura(self) -> str:
        disponivel = _LARGURA - 4
        plain, colorida = self.plain, self.colorida
        if len(plain) > disponivel:
            plain = plain[: disponivel - 1] + "…"
            colorida = plain
        pad = disponivel - len(plain)
        return f"│ {colorida}{' ' * pad} │"


def _titulo(texto: str) -> str:
    disponivel = _LARGURA - 2
    pad_total = disponivel - len(texto)
    esq = pad_total // 2
    dir_ = pad_total - esq
    return f"│{' ' * esq}{_NEGRITO}{_CIANO}{texto}{_RESET}{' ' * dir_}│"


def _fmt_valor(x: float) -> str:
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def _preview(valores: list[float], largura_max: int) -> str:
    if largura_max < 5:
        return "[…]"
    itens = [_fmt_valor(x) for x in valores]
    texto = "[" + ", ".join(itens) + "]"
    if len(texto) <= largura_max:
        return texto
    for n in range(len(itens) - 1, 0, -1):
        candidato = "[" + ", ".join(itens[:n]) + f", … +{len(itens) - n}]"
        if len(candidato) <= largura_max:
            return candidato
    return "[…]"


def _formatar_dataset(ds: Dataset) -> str:
    h = "─" * (_LARGURA - 2)
    metricas = [
        _Metrica("Entradas", ds.num_entradas),
        _Metrica("Saídas", ds.num_saidas),
        _Metrica("Amostras de treino", len(ds.treino)),
        _Metrica("Amostras de teste", len(ds.teste)),
        _Metrica("Total de amostras", len(ds.treino) + len(ds.teste)),
    ]
    lr = max(len(m.rotulo) for m in metricas)

    partes = [f"╭{h}╮", _titulo("DATASET INFO"), f"├{h}┤"]
    partes += [_Linha.de_metrica(m, lr).na_moldura() for m in metricas]

    if ds.treino:
        amostra = ds.treino[0]
        partes.append(f"├{h}┤")
        partes.append(_Linha.destaque("Exemplo (treino[0])").na_moldura())
        partes.append(_Linha.de_amostra("entrada", amostra.entrada).na_moldura())
        partes.append(_Linha.de_amostra("esperado", amostra.esperado).na_moldura())

    partes.append(f"╰{h}╯")
    return "\n".join(partes)
