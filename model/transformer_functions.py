import os
import json

def save_to_json(debug_data, base_name="DebugOutput", dir_path="slides/vars"):
    # Cria o diretório se ele não existir
    os.makedirs(dir_path, exist_ok=True)

    # Gera um nome de arquivo único
    i = 0
    while True:
        filename = os.path.join(dir_path, f"{base_name}_{i}.json")
        if not os.path.exists(filename):
            break
        i += 1

    # Salva o JSON
    with open(filename, "w") as f:
        json.dump(debug_data, f, indent=2)

        f.flush()
        os.fsync(f.fileno())
    print(f"Debug info saved to {filename}")








