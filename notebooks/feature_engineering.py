
# Protótipo do módulo de Feature Engineering
# Esse código será incorporado ao MakeFeatures
# As configurações serão incorporadas ao ConfigMake


from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Timedelta, Timestamp


def get_fpd30_target(df_transacional: DataFrame, data_corte: Timestamp) -> DataFrame:
    """
    Cria a variável alvo FPD30 (First Payment Default 30 dias).
    """
    df_fpd: DataFrame = df_transacional[df_transacional['num_parcela'] == 1].copy()
    df_fpd['data_limite_fpd30'] = df_fpd['data_vencimento'] + Timedelta(days=30)

    df_fpd = df_fpd[df_fpd['data_limite_fpd30'] <= data_corte].copy()

    dias_atraso: pd.Series = (df_fpd['data_pagamento'] - df_fpd['data_vencimento']).dt.days
    df_fpd['dias_atraso'] = dias_atraso.fillna(999).astype(int)
    df_fpd['target'] = np.where(df_fpd['dias_atraso'] > 30, 1, 0)

    return df_fpd

def custom_binning_rules(df, col_name, rules, zero_condition=None):
    """
    Aplica regras de binning customizado.
    
    Parameters
    ----------
    df : pd.DataFrame
    col_name : str
        Coluna numérica original.
    rules : list[tuple(str, float, float)]
        Lista (nome_do_bin, limite_inferior, limite_superior].
        O teste é low < x ≤ high.
    zero_condition : callable, optional
        Função booleana que define o bin 'zero'
        (ex.: lambda x: x==0 or pd.isna(x)).
    """
    out = []
    for v in df[col_name]:
        if zero_condition is not None and zero_condition(v):
            out.append("zero")
            continue

        placed = False
        for name, low, high in rules:
            if low < v <= high:
                out.append(name)
                placed = True
                break
        if not placed:
            out.append("outlier")         # caso fuja de todos os intervalos
    return pd.Series(out, name=f"{col_name}_bin").astype("category")


def calcular_historico_ate_data(df_parcelas: DataFrame, df_contratos_target: DataFrame) -> DataFrame:
    """
    Calcula features históricas de pagamento para cada contrato, incluindo
    atrasos e adiantamentos.
    """
    resultados: List[Dict[str, Any]] = []

    for _, contrato_atual in df_contratos_target.iterrows():
        cliente: int = contrato_atual['identificador_cliente']
        data_originacao_atual: Timestamp = contrato_atual['data_originacao']
        id_contrato_atual: int = contrato_atual['id_contrato']

        contratos_anteriores = df_contratos_target[
            (df_contratos_target['identificador_cliente'] == cliente) &
            (df_contratos_target['data_originacao'] < data_originacao_atual)
        ]

        if len(contratos_anteriores) == 0:
            features = {
                'identificador_cliente': cliente, 'id_contrato': id_contrato_atual, 'data_originacao': data_originacao_atual,
                'qtd_contratos_anteriores': 0, 'cliente_novo': 1
            }
        else:
            parcelas_historicas = df_parcelas[
                (df_parcelas['identificador_cliente'] == cliente) &
                (df_parcelas['id_contrato'].isin(contratos_anteriores['id_contrato'])) &
                (df_parcelas['data_vencimento'] < data_originacao_atual)
            ].copy()

            if len(parcelas_historicas) > 0:
                parcelas_historicas['atraso_dias'] = (parcelas_historicas['data_pagamento'] - parcelas_historicas['data_vencimento']).dt.days
                parcelas_historicas['atrasou'] = parcelas_historicas['atraso_dias'].fillna(9999) > 0
                
                # Lógica para adiantamento
                parcelas_historicas['antecipou'] = (parcelas_historicas['data_pagamento'] < parcelas_historicas['data_vencimento']).fillna(False) 

                total_parcelas = len(parcelas_historicas)
                qtd_atrasos = parcelas_historicas['atrasou'].sum()
                qtd_antecipacoes = parcelas_historicas['antecipou'].sum() 
                max_atraso = parcelas_historicas['atraso_dias'].max() if qtd_atrasos > 0 else 0

                features = {
                    'identificador_cliente': cliente, 'id_contrato': id_contrato_atual, 'data_originacao': data_originacao_atual,
                    'cliente_novo': 0, 'qtd_contratos_anteriores': len(contratos_anteriores),
                    'total_parcelas_historicas': total_parcelas,
                    'qtd_atrasos_historicos': qtd_atrasos,
                    'qtd_antecipacoes_historicas': qtd_antecipacoes, 
                    'taxa_atraso_historica': qtd_atrasos / total_parcelas if total_parcelas > 0 else 0,
                    'taxa_antecipacao_historica': qtd_antecipacoes / total_parcelas if total_parcelas > 0 else 0,
                    'max_atraso_dias_historico': max_atraso,
                    'media_valor_financiado_anterior': contratos_anteriores['valor_financiado'].mean(),
                    'max_valor_financiado_anterior': contratos_anteriores['valor_financiado'].max(),
                    'media_prazo_anterior': contratos_anteriores['prazo'].mean(),
                    'max_prazo_anterior': contratos_anteriores['prazo'].max()
                }
            else:
                # Caso não hajam parcelas históricas, as features de adiantamento também são zeradas.
                features = {
                    'identificador_cliente': cliente, 'id_contrato': id_contrato_atual, 'data_originacao': data_originacao_atual,
                    'cliente_novo': 0, 'qtd_contratos_anteriores': len(contratos_anteriores),
                    'total_parcelas_historicas': 0,
                    'media_valor_financiado_anterior': contratos_anteriores['valor_financiado'].mean(),
                    'max_valor_financiado_anterior': contratos_anteriores['valor_financiado'].max(),
                    'media_prazo_anterior': contratos_anteriores['prazo'].mean(),
                    'max_prazo_anterior': contratos_anteriores['prazo'].max()
                }

        for key in features:
            if pd.isna(features[key]):
                features[key] = 0
        resultados.append(features)

    return pd.DataFrame(resultados)


def calcular_features_temporais(df_contratos: DataFrame) -> DataFrame:
    """
    Calcula features de recência e frequência de contratos de um cliente.
    """
    resultados: List[Dict[str, Any]] = []

    for _, contrato_atual in df_contratos.iterrows():
        cliente: int = contrato_atual['identificador_cliente']
        data_atual: Timestamp = contrato_atual['data_originacao']
        id_contrato_atual: int = contrato_atual['id_contrato']

        contratos_anteriores = df_contratos[
            (df_contratos['identificador_cliente'] == cliente) &
            (df_contratos['data_originacao'] < data_atual)
        ].sort_values('data_originacao')

        if len(contratos_anteriores) > 0:
            ultimo_contrato: Timestamp = contratos_anteriores['data_originacao'].max()
            dias_desde_ultimo: int = (data_atual - ultimo_contrato).days
            primeiro_contrato: Timestamp = contratos_anteriores['data_originacao'].min()
            dias_historico: int = (ultimo_contrato - primeiro_contrato).days

            features_temp = {
                'identificador_cliente': cliente, 'id_contrato': id_contrato_atual,
                'dias_desde_ultimo_contrato': dias_desde_ultimo,
                'dias_como_cliente': dias_historico + dias_desde_ultimo
            }
        else:
            features_temp = {
                'identificador_cliente': cliente, 'id_contrato': id_contrato_atual,
                'dias_desde_ultimo_contrato': 9999, 'dias_como_cliente': 0
            }
        resultados.append(features_temp)

    return pd.DataFrame(resultados)


def calcular_valores_em_aberto(df_parcelas: DataFrame, df_contratos_target: DataFrame) -> DataFrame:
    """
    Calcula features de saldo devedor e parcelas em aberto de um cliente.
    """
    resultados: List[Dict[str, Any]] = []

    for _, contrato_atual in df_contratos_target.iterrows():
        cliente: int = contrato_atual['identificador_cliente']
        data_originacao_atual: Timestamp = contrato_atual['data_originacao']
        id_contrato_atual: int = contrato_atual['id_contrato']

        contratos_anteriores = df_contratos_target[
            (df_contratos_target['identificador_cliente'] == cliente) &
            (df_contratos_target['data_originacao'] < data_originacao_atual)
        ]

        if len(contratos_anteriores) == 0:
            features_aberto = {
                'identificador_cliente': cliente, 'id_contrato': id_contrato_atual, 'data_originacao': data_originacao_atual,
                'qtd_parcelas_em_aberto': 0, 'valor_total_em_aberto': 0
            }
        else:
            parcelas_em_aberto = df_parcelas[
                (df_parcelas['identificador_cliente'] == cliente) &
                (df_parcelas['id_contrato'].isin(contratos_anteriores['id_contrato'])) &
                ((df_parcelas['valor_pago'].isna()) | (df_parcelas['valor_pago'] < df_parcelas['valor_parcela'] * 0.9))
            ].copy()

            if len(parcelas_em_aberto) > 0:
                parcelas_em_aberto['saldo_devedor'] = parcelas_em_aberto['valor_parcela'] - parcelas_em_aberto['valor_pago'].fillna(0)
                valor_total_aberto = parcelas_em_aberto['saldo_devedor'].sum()

                features_aberto = {
                    'identificador_cliente': cliente, 'id_contrato': id_contrato_atual, 'data_originacao': data_originacao_atual,
                    'qtd_parcelas_em_aberto': len(parcelas_em_aberto),
                    'valor_total_em_aberto': valor_total_aberto
                }
            else:
                features_aberto = {
                    'identificador_cliente': cliente, 'id_contrato': id_contrato_atual, 'data_originacao': data_originacao_atual,
                    'qtd_parcelas_em_aberto': 0, 'valor_total_em_aberto': 0
                }
        resultados.append(features_aberto)

    return pd.DataFrame(resultados)

def main():
    """
    Função principal que orquestra todo o processo
    """
    print("Iniciando o processo de engenharia de features...")
    try:
        # Tenta carregar o arquivo do caminho especificado
        data: DataFrame = pd.read_csv("../data/dev/sample_case.csv")
    except FileNotFoundError:
        print("Verificar arquivo")
        return

    date_cols: List[str] = ["data_originacao", "data_vencimento", "data_pagamento"]
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    print("Gerando a variável target FPD30...")
    data_corte: Timestamp = pd.to_datetime("2025-05-12") # Data de corte inferida da base -> vai ser parâmetro da ConfigMake
    data_fpd30: DataFrame = get_fpd30_target(data, data_corte)
    
    if data_fpd30.empty:
        print("Nenhum contrato maduro encontrado para a data de corte.")
        return
        
    data_fpd30["LGD"] = 0.85 # Assumption para crediário pessoal sem garantia 
    data_fpd30["custo_fn"] = data_fpd30["LGD"] * data_fpd30["valor_financiado"]
    data_fpd30["custo_fp"] = (data_fpd30["valor_parcela"] * data_fpd30["prazo"]) - data_fpd30["valor_financiado"]

    # Clipando (Tem no FP mas não tem no FN... valores negativos mt baixo, não vai afetar)
    data_fpd30['custo_fn'] = data_fpd30['custo_fp'].clip(lower=0)
    data_fpd30['custo_fp'] = data_fpd30['custo_fn'].clip(lower=0)

    # Clipando os altos (prone to Overfitting ... )
    limiar_custo_fn = data_fpd30['custo_fn'].quantile(0.95) # -> pode ser incluido no config depois
    limiar_custo_fp = data_fpd30['custo_fp'].quantile(0.95) # -> pode ser inlcuido no config depois

    # Clientes de alto valor
    data_fpd30['flg_alto_valor'] = (
    (data_fpd30['custo_fn'] > limiar_custo_fn) | 
    (data_fpd30['custo_fp'] > limiar_custo_fp)).astype(int)

    # Clip
    data_fpd30['custo_fn'] = data_fpd30['custo_fn'].clip(upper=limiar_custo_fn)
    data_fpd30['custo_fp'] = data_fpd30['custo_fp'].clip(upper=limiar_custo_fp)

    print("Calculando features históricas, temporais e de valores em aberto...")
    df: DataFrame = data.copy()
    df_model: DataFrame = data_fpd30

    features_historicas: DataFrame = calcular_historico_ate_data(df, df_model)
    features_temporais: DataFrame = calcular_features_temporais(df_model)
    features_valores_aberto: DataFrame = calcular_valores_em_aberto(df, df_model)

    print("Unindo todas as features em um único DataFrame...")
    df_final: DataFrame = (
        df_model
        .merge(features_historicas.drop(columns=['data_originacao'], errors='ignore'), on=['identificador_cliente', 'id_contrato'], how='left')
        .merge(features_temporais, on=['identificador_cliente', 'id_contrato'], how='left')
        .merge(features_valores_aberto.drop(columns=['data_originacao'], errors='ignore'), on=['identificador_cliente', 'id_contrato'], how='left')
    )

    numeric_columns: List[str] = df_final.select_dtypes(include=np.number).columns.tolist()
    df_final[numeric_columns] = df_final[numeric_columns].fillna(0)

    print("Aplicando transformação em features cíclicas...")
    df_final['dia_semana_orig'] = df_final['data_originacao'].dt.dayofweek
    df_final['mes_orig'] = df_final['data_originacao'].dt.month
    df_final['trimestre_orig'] = df_final['data_originacao'].dt.quarter

    df_final['dia_semana_sin'] = np.sin(2 * np.pi * df_final['dia_semana_orig'] / 7)
    df_final['dia_semana_cos'] = np.cos(2 * np.pi * df_final['dia_semana_orig'] / 7)
    df_final['mes_sin'] = np.sin(2 * np.pi * df_final['mes_orig'] / 12)
    df_final['mes_cos'] = np.cos(2 * np.pi * df_final['mes_orig'] / 12)
    df_final['trimestre_sin'] = np.sin(2 * np.pi * df_final['trimestre_orig'] / 4)
    df_final['trimestre_cos'] = np.cos(2 * np.pi * df_final['trimestre_orig'] / 4)

    df_final.drop(columns=['dia_semana_orig', 'mes_orig', 'trimestre_orig'], inplace=True)


    #  dicionário de regras -> vai entrar na config
    rules_dict = {
        "qtd_contratos_anteriores": [
            ("1–2", 0, 2),
            ("3+", 2, 50),
        ],
        
        "total_parcelas_historicas": [
            ("1–2", 0, 2),
            ("3",   2, 3),
            ("4",   3, 4),
            ("5–7", 4, 7),
            ("8+",  7, 50),
        ],
        
        "qtd_atrasos_historicos": [
            ("1–2", 0, 2),
            ("3",   2, 3),
            ("4+",  3, 25),
        ],
        
        "qtd_antecipacoes_historicas": [
            ("1–2", 0, 2),
            ("3",   2, 3),
            ("4–6", 3, 6),
            ("7+",  6, 50),
        ],
        
        "taxa_atraso_historica": [
            ("0–25 %",   0,    0.25),
            ("25–50 %",  0.25, 0.50),
            ("50–75 %",  0.50, 0.75),
            ("> 75 %",   0.75, 1.00),
        ],
        
        "taxa_antecipacao_historica": [
            ("7–50 %",   0.06, 0.50),
            ("50–80 %",  0.50, 0.80),
            ("80–100 %", 0.80, 1.00),
        ],
        
        "max_atraso_dias_historico": [
            ("≤ 2 d",    0,   2),
            ("3–4 d",    2,   4),
            ("5–8 d",    4,   8),
            ("9–18 d",   8,  18),
            ("> 18 d",  18, 400),
        ],
        
        "media_valor_financiado_anterior": [
            ("≤ 120",    0,   120),
            ("120–180", 120,  180),
            ("180–250", 180,  250),
            ("250–400", 250,  400),
            ("> 400",   400, 1e6),
        ],
        
        "max_valor_financiado_anterior": [
            ("≤ 130",    0,   130),
            ("130–190", 130,  190),
            ("190–280", 190,  280),
            ("280–450", 280,  450),
            ("> 450",   450, 1e6),
        ],
        
        "media_prazo_anterior": [
            ("1–3 meses", 0, 3),
            ("4 meses",   3, 4),
            ("5–14 mês",  4, 14),
        ],
        
        "max_prazo_anterior": [
            ("1–3 meses", 0, 3),
            ("4 meses",   3, 4),
            ("5–14 mês",  4, 14),
        ],
        
        "dias_desde_ultimo_contrato": [
            ("≤ 20 d",    0,    20),
            ("21–50 d",  20,    50),
            ("51–100 d", 50,   100),
            ("101–180 d",100,  180),
            ("> 180 d", 180, 1e4),
        ],
        
        "dias_como_cliente": [
            ("≤ 43 d",   0,   43),
            ("44–100 d",43,  100),
            ("101–177 d",100, 177),
            ("178–290 d",177, 290),
            ("> 290 d", 290, 600),
        ],
        
        "qtd_parcelas_em_aberto": [
            ("1–2", 0, 2),
            ("3",   2, 3),
            ("4",   3, 4),
            ("5+",  4, 40),
        ],
        
        "valor_total_em_aberto": [
            ("≤ 10",     0,     10.07),
            ("10–30",   10.07,  30),
            ("30–75",   30,     75),
            ("75–175",  75,    175),
            ("> 175",  175,   6_000),
        ],
    }

    zero_cond = lambda x: (x == 0) or pd.isna(x)

    for col, rules in rules_dict.items():
        df_final[f"{col}_bin"] = custom_binning_rules(df_final, col, rules, zero_condition=zero_cond)
    
    # Features adicionais / pequenas correções
    df_final['proporcao_parcela_financiado'] = df_final['valor_parcela'] / df_final['valor_financiado']
    df_final['custo_efetivo_total_proxy'] = (df_final['valor_parcela'] * df_final['prazo']) / df_final['valor_financiado']
    data_fpd30['tipo_cliente'] = data_fpd30['tipo_cliente'].replace('Erro', 'Outros')
    df_final = df_final.drop(columns = rules_dict.keys()) 
    df_final = df_final.drop(columns = [
        "data_vencimento", 
        "data_pagamento", 
        "dias_atraso",
        "LGD",
        "valor_pago",
        "identificador_cliente",
        "data_limite_fpd30"
    ]) 

    print(f"\nDimensões do DataFrame final: {df_final.shape}")
    df_final.to_csv("../data/dev/abt_modelo_fpd30_v3.csv", index=False)

if __name__ == "__main__":
    main()
