import subprocess
import time
import logging
from datetime import datetime

# Configuração de log
logging.basicConfig(
    filename="sync_time.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

SYNC_INTERVAL = 0 * 0  # 30 minutos em segundos

def sync_windows_time():
    try:
        logging.info("Iniciando sincronização do horário do Windows")

        # Força atualização do serviço de horário
        subprocess.run(
            ["w32tm", "/resync"],
            capture_output=True,
            text=True,
            check=True
        )

        logging.info("Horário sincronizado com sucesso")

    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao sincronizar horário: {e.stderr}")

if __name__ == "__main__":
    logging.info("Script de sincronização iniciado")
    sync_windows_time()
    time.sleep(SYNC_INTERVAL)
    print('Hora sincronizada com sucesso!')
