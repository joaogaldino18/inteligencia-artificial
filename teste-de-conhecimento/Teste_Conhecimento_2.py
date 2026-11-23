import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Leitura dos pesos
# ---------------------------------------------------------------------------

def carregar_rede(arquivo="pesos_rede.json"):
    """Carrega os pesos da rede e converte os valores para NumPy."""
    with open(arquivo, encoding="utf-8") as f:
        cfg = json.load(f)

    chaves = ["W1", "b1", "W2", "b2", "W3", "b3", "normalizacao_divisor"]

    for chave in chaves:
        cfg[chave] = np.asarray(cfg[chave], dtype=float)

    return cfg


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def normalizar(entrega_bruta, divisores):
    """Normaliza os atributos da entrega utilizando seus divisores."""
    valores = np.asarray(entrega_bruta, dtype=float)
    escala = np.asarray(divisores, dtype=float)
    x = valores / escala
    return x


def relu(z):
    """Aplica a função ReLU."""
    return np.where(z > 0, z, 0)


def tanh(z):
    """Aplica a tangente hiperbólica."""
    return np.tanh(z)


def softmax(z):
    """Calcula as probabilidades de saída da rede."""
    z = np.asarray(z, dtype=float)
    exp_z = np.exp(z - np.max(z))
    soma = exp_z.sum()
    return exp_z / soma


def camada(W, b, x, ativacao):
    """Calcula a saída de uma camada da rede."""
    resultado = np.dot(W, x) + b
    saida = ativacao(resultado)
    return resultado, saida


def propagar(rede, x):
    """Realiza a propagação direta pelas camadas."""
    z1, h1 = camada(rede["W1"], rede["b1"], x, relu)
    z2, h2 = camada(rede["W2"], rede["b2"], h1, tanh)
    z3, p = camada(rede["W3"], rede["b3"], h2, softmax)

    resultado = {
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "z3": z3,
        "p": p
    }

    return resultado


def classificar(rede, x):
    """Identifica a classe com maior probabilidade."""
    resultado = propagar(rede, x)
    p = resultado["p"]
    indice = np.argmax(p)

    nome = rede["classes"][indice]
    probabilidade = float(p[indice])

    return nome, probabilidade


# ---------------------------------------------------------------------------
# Teste de validação
# ---------------------------------------------------------------------------

def _confere(nome, obtido, esperado, tol=1e-3):
    ok = np.allclose(np.asarray(obtido, dtype=float),
                     np.asarray(esperado, dtype=float), atol=tol)

    print(f"[{'OK' if ok else 'FALHOU'}] {nome}")

    if not ok:
        print("    esperado:", np.round(esperado, 4))
        print("    obtido: ", np.round(obtido, 4))


def validar(rede):
    print("=" * 60)
    print("VALIDAÇÃO: x = (0.60, 0.40, 0.50, 0.20)")
    print("=" * 60)

    x = np.array([0.60, 0.40, 0.50, 0.20])
    r = propagar(rede, x)

    _confere("z1", r["z1"], [1.602, -0.417, 1.433, -1.171, -0.521, 1.274])
    _confere("h1", r["h1"], [1.602, 0.000, 1.433, 0.000, 0.000, 1.274])
    _confere("z2", r["z2"], [2.5760, -0.9262, 4.1550, 0.0951])
    _confere("h2", r["h2"], [0.9885, -0.7288, 0.9995, 0.0948])
    _confere("z3", r["z3"], [-3.5755, 4.4592, -1.0141])
    _confere("p", r["p"], [0.0003, 0.9955, 0.0042])

    classe, prob = classificar(rede, x)
    print(f"Classe prevista: {classe} ({100 * prob:.2f}%) -> esperado: SEDEX (99.55%)")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        rede = carregar_rede()
        validar(rede)

        np.set_printoptions(precision=4, suppress=True)

        chaves_pesos = ["W1", "b1", "W2", "b2", "W3", "b3"]
        total = sum(rede[k].size for k in chaves_pesos)

        print("\nNúmero de parâmetros:", total)

        df = pd.read_csv("entregas_teste_rede.csv")
        print("\nEntregas de teste:")

        for _, linha in df.iterrows():
            dados = [
                linha.distancia_km,
                linha.peso_kg,
                linha.urgencia,
                linha.valor_declarado
            ]

            x = normalizar(dados, rede["normalizacao_divisor"])
            classe, prob = classificar(rede, x)

            print(f"  entrega {linha.id_entrega:2d}: prevista = {classe:8s} ({100 * prob:6.2f}%)"
                  f" | real = {linha.modalidade_real}")

    except NotImplementedError as e:
        print(f"\nFunção ainda não implementada: {e}()")
