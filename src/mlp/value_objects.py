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


class DataChoiceEnum(Enum):
    OR = "OR"
    AND = "AND"
    XOR = "XOR"
    CARACTERES = "CARACTERES"
    CARACTERES_COMPLETO = "CARACTERES_COMPLETO"
