# Guia de instalação e execução — VS Code (Windows)

Como instalar e rodar o projeto do zero. Todos os comandos são para o **PowerShell** do VS Code, dentro da pasta `ProjetoTechChallenge2`.

## 1. Pré-requisitos

- **Python 3.11** ([python.org](https://www.python.org/downloads/windows/)) — na instalação, marque **"Add python.exe to PATH"**. Confira com `py -3.11 --version`.
- **VS Code** com as extensões **Python** e **Jupyter** da Microsoft (`Ctrl+Shift+X`).
- **Chave da OpenAI**, apenas para a parte de LLM (passo 7).

## 2. Abrir o projeto e criar o ambiente

No VS Code: **File → Open Folder** → pasta `ProjetoTechChallenge2` → **Terminal → New Terminal**.

```powershell
py -3.11 -m venv .venv                       # cria o ambiente (uma vez)
.\.venv\Scripts\Activate.ps1                 # ativa (a cada terminal novo)
python -m pip install -r requirements.txt    # instala as bibliotecas (alguns minutos)
```

O terminal passa a mostrar `(.venv)`. Se a ativação der erro de "execução de scripts desabilitada", rode uma vez `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` e ative de novo.

Nos notebooks, escolha o mesmo ambiente: **Select Kernel → Python Environments → `.venv`**.

## 3. Treinar os modelos (notebook do algoritmo genético)

Abra **`diagnosticos_algoritmo_genetico.ipynb`** → **Run All** (cerca de 2 minutos).

Ele faz a análise exploratória, treina os modelos originais, roda os 3 experimentos do algoritmo genético nos 3 modelos, compara os resultados, **exporta `modelo_pipeline_genetico.pkl`** e salva gráficos e números em `resultados/`.

> O modelo e os resultados já vêm prontos no repositório: este passo serve para reproduzi-los (a semente fixa gera sempre os mesmos números).

## 4. Rodar os testes

```powershell
python -m pytest
```

Esperado: `78 passed, 6 skipped`. Os 6 pulados chamam a OpenAI de verdade e são opcionais (passo 7). Os testes não precisam da chave nem da API rodando.

## 5. Subir a API

```powershell
uvicorn main:app
```

Espere `Modelo carregado com sucesso` e `Uvicorn running on http://127.0.0.1:8000`. Deixe o terminal aberto (`Ctrl+C` para parar).

| Rota | Para quê |
|---|---|
| `GET /health/live` · `GET /health/ready` | a API está no ar? o modelo certo foi carregado? |
| `GET /metrics` | métricas Prometheus (requisições, predições, explicações) |
| `POST /predict` | classifica os 4 valores do exame |
| `POST /explicar` | texto do exame → classificação + explicação da LLM (passo 7) |

## 6. Testar a predição pelo Swagger

Abra **http://127.0.0.1:8000/docs** → **POST /predict** → *Try it out* → *Execute* com:

```json
{"concave_points_mean": 0.1471, "concave_points_worst": 0.2654, "perimeter_worst": 184.6, "radius_worst": 25.38}
```

Esperado: `"label": "Maligno"`. Com `0.0, 0.0, 59.16, 9.456` → `"Benigno"`. Com um valor negativo → erro **422** explicando o campo.

## 7. Configurar a OpenAI e testar a explicação

Crie o `.env` e troque `sua_chave_aqui` pela sua chave (o `.env` não vai para o Git):

```powershell
Copy-Item .env.example .env
notepad .env
```

Reinicie a API (`Ctrl+C` e `uvicorn main:app`) e use uma das opções:

- **Pela API:** no Swagger, **POST /explicar** → *Try it out* →
  `{"texto_exame": "Exame sintético: pontos côncavos (média) 0,121; pontos côncavos (pior valor) 0,245; perímetro (pior valor) 165,3; raio (pior valor) 24,8."}`
  → a resposta traz os valores extraídos, a classificação do modelo e a explicação. Se faltar uma medida, o status é `revisao_necessaria`.
- **Pelo notebook:** com a API rodando, abra **`assistente_laudo_medico.ipynb`** → **Run All**. Ele mostra os prompts, processa um exame e avalia 5 cenários, salvando em `resultados/avaliacao_llm.json`.
- **Testes reais (opcional):** `$env:RUN_OPENAI_TESTS = "1"; python -m pytest tests/test_openai_real.py -v`

Cada exame explicado usa 2 chamadas ao `gpt-4.1-mini` (frações de centavo).

## 8. Problemas frequentes

| Problema | Solução |
|---|---|
| `py` não é reconhecido | Reinstale o Python marcando *Add python.exe to PATH*. |
| `ModuleNotFoundError` (ex.: `fastapi`, `modelos`) | Ative o `.venv`, selecione-o como kernel e abra a **pasta** do projeto (não só o arquivo). |
| Instalação lenta ou erro de caminho longo | Pause o OneDrive durante a instalação ou use uma pasta curta, como `C:\fiap\fase2`. |
| Porta 8000 em uso | Feche a outra API ou use `uvicorn main:app --port 8001`. |
| API: modelo não encontrado ou sem `scaler` | Rode o notebook do passo 3 e não defina `MODEL_PATH` para outro arquivo. |
| `/explicar` responde 503 / `OPENAI_API_KEY não configurada` | Configure o `.env` (passo 7) e reinicie a API. |
| OpenAI `401`, `429` ou `model_not_found` | Chave inválida, sem créditos ou modelo indisponível: confira a conta ou troque `OPENAI_MODEL` no `.env`. |
