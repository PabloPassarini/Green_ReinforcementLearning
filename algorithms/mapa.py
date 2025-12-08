import pandas as pd
import tsplib95
import matplotlib.pyplot as plt
from pathlib import Path
import os
def gerar_grafico(coords, sol, info):
    x = list()
    y = list()
    for id, coord in coords.items():
        x.append(coord[0])
        y.append(coord[1])

    plt.figure(figsize=(10,8))
    plt.scatter(x, y, color='blue', marker='o', label='Points')

    x_rota = list()
    y_rota = list()

    for p in sol:
        coord = coords.get(p+1)
        if coord != None:
            x_rota.append(coord[0])
            y_rota.append(coord[1])
    
    plt.plot(x_rota, y_rota, linewidth=2, alpha=0.7, label='Route')
    plt.title(f'Instanse: {info[0]} | Run: {sol[2]} | Gamma: {info[1]}')
    plt.grid(True)
    plt.legend()
    
    pasta = Path(
        "plots/Pablo"
    )

    plt.savefig(
        os.path.join(pasta, f"Instanse_{info[0]}_Run_{sol[2]}_Gamma_{str(info[1]).replace('.', 'p')}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    #plt.show()



summary_path = Path(
        "results/q-learning/20251205T131931+0000/master_summary.csv"
    )
berlin = r'F:\Projetos\Programacao\Green_ReinforcementLearning\instances\berlin52.tsp'









df = pd.read_csv(summary_path, decimal='.', sep=',')

df = df[df['instance'].isin(['kroA100.tsp', 'berlin52.tsp'])]
print(df)
for id, sol in df.iterrows():

    rota = str(sol['BestPath']).replace(' ', '')
    rota = rota.split('->')
    rota = [int(p) for p in rota]
    
    tsp_file = Path("instances") / sol['instance']
    problema = tsplib95.load(tsp_file)
    dict_ordenado = dict(sorted(problema.node_coords.items()))
    info = [sol['instance'], sol['gamma'], sol['run_index']]
    gerar_grafico(dict_ordenado, rota, info)
  




