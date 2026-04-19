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


@dataclass
class ResultadoTeste:
    """Resultado de uma amostra de teste depois do forward."""

    entrada: list[float]
    esperado_raw: list[float]  # vetor cru esperado (ex: [1, -1, -1, ...])
    saida_raw: list[float]  # vetor cru produzido pela rede
    classe_predita: int  # indice (multiclasse) ou -1/+1 (binario)
    classe_esperada: int
    acerto: bool


class DataChoiceEnum(Enum):
    OR = "OR"
    AND = "AND"
    XOR = "XOR"
    CARACTERES_REDUZIDO = "CARACTERES_REDUZIDO"
    CARACTERES_COMPLETO = "CARACTERES_COMPLETO"
