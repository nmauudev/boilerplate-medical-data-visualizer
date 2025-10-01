import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importa los datos
df = pd.read_csv('medical_examination.csv')

# 2. Agrega la columna 'overweight'
# Calcula el BMI. La altura está en cm, se convierte a metros (dividiendo por 100).
# Luego aplica la fórmula: peso (kg) / altura (m)^2
df['overweight'] = (df['weight'] / (df['height'] / 100)**2).apply(lambda x: 1 if x > 25 else 0)

# 3. Normaliza los datos
# Si el valor es 1 (normal), se convierte en 0 (bueno).
# Si el valor es mayor que 1 (anormal), se convierte en 1 (malo).
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Dibuja el Gráfico Categórico
def draw_cat_plot():
    # 5. Crea un DataFrame para el cat plot usando pd.melt
    # Se transforman las columnas de features a un formato largo (long format).
    # 'cardio' es el identificador, y las otras son las variables a medir.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupa y reformatea los datos para dividirlos por 'cardio'
    # Se agrupa por 'cardio', 'variable' y 'value' para contar las ocurrencias.
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()

    # 7. Dibuja el catplot usando sns.catplot()
    # 'col="cardio"' divide el gráfico en dos, uno para cardio=0 y otro para cardio=1.
    # 'kind="bar"' especifica que queremos un gráfico de barras.
    # 'hue="value"' colorea las barras según el valor (0 o 1).
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8. Guarda la figura
    fig.savefig('catplot.png')
    return fig

# 10. Dibuja el Heat Map
def draw_heat_map():
    # 11. Limpia los datos
    # Se filtran los datos incorrectos o atípicos.
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calcula la matriz de correlación
    corr = df_heat.corr()

    # 13. Genera una máscara para el triángulo superior
    # Esto es para evitar mostrar información duplicada en el heatmap.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configura la figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Dibuja el heatmap con seaborn
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, square=True, cbar_kws={"shrink": .5}, ax=ax)

    # 16. Guarda la figura
    fig.savefig('heatmap.png')
    return fig