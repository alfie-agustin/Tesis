import pandas as pd
import numpy as np


def pipeline(n_df):
    ## na

    # ph en suelo se remplaza por el ph de la localidad
    ph_suelo = n_df.groupby(["localidad"])["ph"].mean().reset_index().sort_values(by="ph", ascending=False)
    ph_suelo = ph_suelo.rename(columns={"ph": "ph_localidad"})

    n_df = pd.merge(n_df, ph_suelo, on="localidad", how="left")
    n_df["ph"] = np.where(n_df["ph"].isnull(), n_df["ph_localidad"], n_df["ph"])
    n_df.drop(columns=["ph_localidad"], inplace=True)

    # si no hay dato de resiembra se remplaza por un no, ya que no se pued asumir resiembra si no la hay
    n_df["resiembra"] = n_df["resiembra"].fillna("no")

    # remplazo fecha_cosecha con el valor mas cercano de id ya que el id es autocreciente
    n_df["fecha_cosecha"] = n_df["fecha_cosecha"].fillna(method="ffill")

    # se busca y remplaza influencia napa por si se encuentra la influencia en el mismo lote en otro registro
    n_df['influencia_napa'] = n_df.groupby('lote')['influencia_napa'].transform(lambda x:
                                                                                x.fillna('si') if 'si' in x.values else
                                                                                x.fillna('no') if 'no' in x.values else
                                                                                'no')

    # el sistema de siembra se coloca por la moda de la localidad si esta vacio

    modas_por_localidad = n_df.groupby('localidad')['sistema_siembra'].apply(
        lambda x: x.mode()[0] if not x.dropna().empty else 'Valor_por_defecto')
    n_df['sistema_siembra'] = n_df.apply(
        lambda row: modas_por_localidad[row['localidad']] if pd.isnull(row['sistema_siembra']) else row[
            'sistema_siembra'], axis=1)

    # se remplaza na en version biotecnologica como no aplica para los que no son maiz ni trigo para los otros tipos de cultivo y la moda de la version para los que son

    n_df["version_biotecnologica"] = np.where(
        (n_df["version_biotecnologica"].isnull()) & (n_df["cultivo"] != "maiz") & (n_df["cultivo"] != "girasol"),
        "No aplica", "no bt")

    # valores nulos de densidad_sem_m2 se calculca a partir de la densidad sembrada por ha si esta
    n_df["densidad_sem_m2"] = np.where(n_df["densidad_sem_m2"].isnull(), n_df["densidad_sem_ha"] * 10000,
                                       n_df["densidad_sem_m2"])
    # Reemplaza los valores nulos por la densidad promedio correspondiente a cada lote
    dens_prob = n_df.groupby('lote')['densidad_sem_m2'].apply(lambda x: x.mean())
    n_df['densidad_sem_m2'] = n_df.apply(
        lambda row: dens_prob[row['lote']] if pd.isnull(row['densidad_sem_m2']) else row['densidad_sem_m2'], axis=1)

    n_df.drop(["densidad_sem_ha"], inplace=True, axis=1)
    n_df.drop("nivel_informacion", inplace=True, axis=1)

    # el sistema labranza se cambia por la moda
    n_df["sistema_labranza"] = np.where(n_df["sistema_labranza"].isnull(), "siembra directa", n_df["sistema_labranza"])

    # cambio sistematizado por la moda
    n_df["sistematizado"] = np.where(n_df["sistematizado"].isnull(), "no sistematizado", n_df["sistematizado"])

    n_df.drop("pp_campana", inplace=True, axis=1)  # se hara un data augmentation posterior

    # se cambia por el ciclo más repetido para un cultivo

    n_df["version_biotecnologica"] = np.where(
        (n_df["version_biotecnologica"].isnull()) & (n_df["cultivo"] != "maiz") & (n_df["cultivo"] != "girasol"),
        "No aplica", "no bt")


    ciclo_mas_repetido = {
        "trigo": "intermedio",
        "soja": "iv largo",
        "maiz": "largo",
        "girasol": "largo",
        "cebada": "intermedio"
    }

    n_df["ciclo"] = n_df["ciclo"].fillna(n_df["cultivo"].map(ciclo_mas_repetido))

    # temprano tardio solo aplica para maiz, si el cultivo no es maiz se pone no aplica, sino se pone moda

    n_df["mes_fecha_cosecha"] = pd.to_datetime(n_df["fecha_cosecha"], format="%m")

    n_df["temprano_tardio"] = np.where(
        (n_df["temprano_tardio"].isnull()) & (n_df["cultivo"] != "maiz"),
        "No aplica",
        np.where(
            (n_df["mes_fecha_cosecha"].notnull()) &
            (n_df["mes_fecha_cosecha"].dt.month > 11),
            "tardio",
            "temprano"
        )
    )

    promedio_localidad = n_df.groupby('localidad')['poblacion_pl_ha'].mean()
    promedio_lote = n_df.groupby('lote')['poblacion_pl_ha'].mean()

    # Función para obtener el promedio específico del lote o de la localidad según el caso
    def obtener_promedio(row):
        if pd.isnull(row['poblacion_pl_ha']):
            lote = row['lote']
            if lote in promedio_lote:
                return promedio_lote[lote]
            else:
                return promedio_localidad[row['localidad']]
        else:
            return row['poblacion_pl_ha']

    # Aplicar la función para reemplazar los valores nulos en poblacion_pl_ha
    n_df["poblacion_pl_ha"] = n_df.apply(obtener_promedio, axis=1)

    # si el suelo tiene salinidad --> voy a usar la moda de la zona para si o no
    n_df["ambiente_salino"].unique()

    amb_sal = n_df.groupby(["localidad", "ambiente_salino"])["id"].count().reset_index().sort_values(by="localidad",
                                                                                                     ascending=True)
    amb_sal.rename(columns={"id": "count"}, inplace=True)

    amb_sal_max = amb_sal.groupby("localidad").first().reset_index()
    n_df = pd.merge(n_df, amb_sal_max, on="localidad", suffixes=('_original', '_max'))
    n_df["ambiente_salino"] = np.where(n_df["ambiente_salino_original"].isnull(), n_df["ambiente_salino_max"],
                                       n_df["ambiente_salino_original"])
    n_df.drop("ambiente_salino_max", inplace=True, axis=1)

    # se elimina porque ya hay una variable que explica lo mismo con nas rempleazados
    n_df.drop(["densidad_kg_ha"], inplace=True, axis=1)

    ## outliers

    n_df = outliers(n_df, "poblacion_pl_ha")
    n_df = outliers(n_df, "distancia_hileras")
    n_df = outliers(n_df, "ph")
    n_df = outliers(n_df, "duracion_campaña")

    return n_df


def outliers(df, var):
    median = df[var].median()

    # Calcular el rango intercuartílico (IQR)
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1

    # Definir los límites para los outliers
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Reemplazar los outliers por la mediana
    df[var] = df[var].apply(lambda x: median if x < lower_limit or x > upper_limit else x)

    return df
