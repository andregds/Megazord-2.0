import pandas as pd
import os

# Caminho base
base_path = r"D:\GOOGLE DRIVE\Material_de_estudo\Python\Megazord - 2.0"

file1 = os.path.join(base_path, "signals_log.csv")       # referência
file2 = os.path.join(base_path, "signals_log--.csv")     # arquivo quebrado
output = os.path.join(base_path, "signals_log--_fixed.csv")

# Lê o arquivo referência para pegar os nomes de colunas
df1 = pd.read_csv(file1, sep=",", engine="python")
colunas = df1.columns.tolist()

# Leitura bruta do segundo arquivo
with open(file2, "r", encoding="utf-8", errors="ignore") as f:
    linhas = f.read().splitlines()

dados = []
for linha in linhas:
    # divide no máximo em 17 vírgulas → gera 18 colunas
    partes = linha.split(",", 17)
    if len(partes) == 18:
        dados.append(partes)
    else:
        # Linha mal formatada, preenche até 18 colunas
        partes = (partes + [""] * 18)[:18]
        dados.append(partes)

# Cria DataFrame padronizado
df2 = pd.DataFrame(dados, columns=colunas)

# Conversão de datas
for col in ["detect_time", "evaluated_at"]:
    df2[col] = pd.to_datetime(df2[col], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")

# Conversão de números
num_cols = ["prob","rsi","ema","vwap","price","volume","entry","stop","target","atr","rr"]
for col in num_cols:
    df2[col] = pd.to_numeric(df2[col], errors="coerce")

# Salva ajustado
df2.to_csv(output, index=False, encoding="utf-8")

print(f"✅ Arquivo corrigido salvo em: {output}")
print(f"📊 Total de linhas: {len(df2)}")
