# Planejamento Amostral
# Esse código será incorporado ao Make -> treinamento do modelo

#In-Time: O conjunto de validação/teste é formado por instâncias do mesmo período de tempo do treino.
#Out-of-Time: O conjunto de validação/teste é formado por instâncias retiradas de um período futuro em relação ao treino 

# Utilizaremos um conjunto de treino (dados originados de 2023-06-01 a 2024-05-31)
# Conjunto de validação para ajuste de hiperparâmetro: (Validação: de 2024-06-01 a 2024-08-31) -> poderia ser In-time dado a limitação dos dados
# Conjunto de teste OOT de 2024-09-01 a 2024-12-31

# Att:
# Para treino e validação optei por usar janela expansiva (walk-forward expansivo)
# Para calibração das probabilidades, dado a limitação do dado, também usaremos o treino + val (In - Time)

from typing import Annotated, Dict, Tuple

import pandas as pd


def time_train_test_split_fpd30(
    df: pd.DataFrame, 
    coluna_data: Annotated[str, "Nome da coluna com a data de originação."]
) -> Dict[str, pd.DataFrame]:
    """
    Divide um DataFrame em conjuntos de treino, validação e teste (out-of-time)
    com base em janelas de tempo cronológicas e contíguas.

    Args:
        df (pd.DataFrame): O DataFrame completo contendo todos os dados.
        coluna_data (str): O nome da coluna de data de originação.

    Returns:
        Dict[str, pd.DataFrame]: Um dicionário contendo os DataFrames de 
                                 'treino', 'validacao' e 'teste_oot'.
    """
    df_copy = df.copy()
    df_copy[coluna_data] = pd.to_datetime(df_copy[coluna_data])
    df_copy = df_copy.sort_values(by=coluna_data).reset_index(drop=True)
    
    data_fim_treino = pd.to_datetime('2024-05-31')
    data_fim_validacao = pd.to_datetime('2024-08-31')

    df_treino = df_copy.loc[df_copy[coluna_data] <= data_fim_treino].copy()

    df_validacao = df_copy.loc[
        (df_copy[coluna_data] > data_fim_treino) &
        (df_copy[coluna_data] <= data_fim_validacao)
    ].copy()

    df_teste_oot = df_copy.loc[df_copy[coluna_data] > data_fim_validacao].copy()

    folds = {
        "treino": df_treino,
        "validacao": df_validacao,
        "teste_oot": df_teste_oot
    }
    
    print("Divisão concluída")
    print(f"  - Registros de Treino: {len(df_treino):,}")
    print(f"  - Registros de Validação: {len(df_validacao):,}")
    print(f"  - Registros de Teste (OOT): {len(df_teste_oot):,}")
    
    return folds

if __name__ == '__main__':

    df = pd.read_csv("../data/dev/abt_modelo_fpd30_v3.csv")
    split_folds = time_train_test_split_fpd30(
        df=df, 
        coluna_data="data_originacao"
    )

    treino_df = split_folds["treino"]
    validacao_df = split_folds["validacao"]
    teste_oot_df = split_folds["teste_oot"]

    print("\n" + "="*50 + "\n")
    
    if not treino_df.empty:
        print(f"Treino: de {treino_df['data_originacao'].min().date()} a {treino_df['data_originacao'].max().date()}")
    
    if not validacao_df.empty:
        print(f"Validação: de {validacao_df['data_originacao'].min().date()} a {validacao_df['data_originacao'].max().date()}")

    if not teste_oot_df.empty:
        print(f"Teste OOT: de {teste_oot_df['data_originacao'].min().date()} a {teste_oot_df['data_originacao'].max().date()}")

    treino_df.to_csv("../data/dev/train.csv")
    validacao_df.to_csv("../data/dev/val.csv")
    teste_oot_df.to_csv("../data/dev/test.csv")