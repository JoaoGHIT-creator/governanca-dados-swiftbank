"""Diagnóstico de qualidade dos metadados do SwiftBank (4ª etapa).

Pensado para rodar no Google Colab: envie o metadados_tabelas_swiftbank.csv
para o ambiente e ajuste CAMINHO_CSV se precisar.

Lê o arquivo mesmo que as descrições com vírgula (ex.: "Faixa do score (baixo,
médio, alto)") venham sem aspas: se a linha tiver mais de 8 campos, os campos
do meio são reunidos de volta na descrição.
"""
import csv
import pandas as pd

CAMINHO_CSV = "metadados_tabelas_swiftbank.csv"
CAMADAS_PERMITIDAS = ["bronze", "silver", "gold"]
CAMPOS_COMPLETUDE = ["descricao", "confidencialidade", "dado_pessoal", "responsavel"]
COLUNAS = ["tabela", "camada", "coluna", "descricao", "tipo_dado",
           "confidencialidade", "dado_pessoal", "responsavel"]
LIMITE_PALAVRAS = 3


def ler_metadados(caminho=CAMINHO_CSV):
    with open(caminho, encoding="utf-8", newline="") as f:
        linhas = list(csv.reader(f))
    cabecalho, dados = linhas[0], linhas[1:]
    assert cabecalho == COLUNAS, f"Cabeçalho inesperado: {cabecalho}"
    ajustadas = []
    for campos in dados:
        if len(campos) > 8:  # vírgula dentro da descrição, sem aspas
            campos = campos[:3] + [",".join(campos[3:len(campos) - 4])] + campos[-4:]
        ajustadas.append(campos)
    df = pd.DataFrame(ajustadas, columns=COLUNAS)
    return df.apply(lambda c: c.str.strip())


def diagnosticar(df):
    # Células vazias contam como nulas
    vazio = df[CAMPOS_COMPLETUDE].replace("", pd.NA).isna()

    # (a) % de completude por campo
    completude = ((1 - vazio.mean()) * 100).round(1).rename("completude_%").reset_index()
    completude.columns = ["campo", "completude_%"]

    # (b) camadas fora da lista permitida
    camada_invalida = ~df["camada"].str.lower().isin(CAMADAS_PERMITIDAS)

    # (c) descrições preenchidas com 3 palavras ou menos
    n_palavras = df["descricao"].str.split().str.len().fillna(0)
    desc_curta = (df["descricao"] != "") & (n_palavras <= LIMITE_PALAVRAS)

    # extra: mesma coluna com classificação de dado pessoal diferente entre tabelas
    divergente = (df.groupby("coluna")["dado_pessoal"]
                  .transform(lambda s: s[s != ""].nunique() > 1))

    # (d) ranking de tabelas: problemas por verificação (6 verificações por linha)
    prob = pd.DataFrame({
        "tabela": df["tabela"],
        "campos_vazios": vazio.sum(axis=1),
        "camada_invalida": camada_invalida.astype(int),
        "descricao_curta": desc_curta.astype(int),
    })
    prob["problemas"] = prob[["campos_vazios", "camada_invalida", "descricao_curta"]].sum(axis=1)
    ranking = (prob.groupby("tabela")
               .agg(colunas=("tabela", "size"),
                    campos_vazios=("campos_vazios", "sum"),
                    camada_invalida=("camada_invalida", "sum"),
                    descricao_curta=("descricao_curta", "sum"),
                    problemas=("problemas", "sum"))
               .reset_index())
    ranking["indice_problemas_%"] = (ranking["problemas"] / (ranking["colunas"] * 6) * 100).round(1)
    ranking = (ranking.sort_values(["indice_problemas_%", "problemas"], ascending=False)
               .reset_index(drop=True))
    ranking.insert(0, "posicao", ranking.index + 1)

    # observação por linha (alimenta o dicionário de dados)
    def observar(i):
        itens = [f"{c} vazio" for c in CAMPOS_COMPLETUDE if vazio.loc[i, c]]
        if camada_invalida[i]:
            itens.append("camada fora da lista (bronze, silver, gold)")
        if desc_curta[i]:
            itens.append("descrição com 3 palavras ou menos")
        if divergente[i]:
            itens.append("dado_pessoal diverge entre tabelas para a mesma coluna")
        return "; ".join(itens)
    df = df.assign(observacao_de_qualidade=[observar(i) for i in df.index])

    resumo = {
        "linhas": len(df),
        "linhas_camada_invalida": int(camada_invalida.sum()),
        "descricoes_curtas": int(desc_curta.sum()),
        "linhas_com_algum_vazio": int(vazio.any(axis=1).sum()),
        "colunas_com_classificacao_divergente": sorted(df.loc[divergente, "coluna"].unique()),
    }
    return completude, resumo, ranking, df, df[desc_curta][["tabela", "coluna", "descricao"]]


if __name__ == "__main__":
    df = ler_metadados()
    completude, resumo, ranking, df_obs, curtas = diagnosticar(df)
    print("Linhas lidas:", resumo["linhas"], "\n")
    print("(a) Completude por campo")
    print(completude.to_string(index=False), "\n")
    print("(b) Linhas com camada fora da lista:", resumo["linhas_camada_invalida"])
    print(df[~df["camada"].isin(CAMADAS_PERMITIDAS)][["tabela", "coluna", "camada"]].to_string(index=False), "\n")
    print("(c) Descrições com 3 palavras ou menos:", resumo["descricoes_curtas"])
    print(curtas.to_string(index=False), "\n")
    print("(d) Ranking de tabelas (pior qualidade primeiro)")
    print(ranking.to_string(index=False), "\n")
    print("Extra: colunas com dado_pessoal divergente entre tabelas:",
          resumo["colunas_com_classificacao_divergente"])
