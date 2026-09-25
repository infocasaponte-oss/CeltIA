# Corpus curado para CeltIA

Este descargador prepara un corpus grande en **D:\\corpus_llm_grande** usando streaming, filtros de calidade, deduplicación exacta e checkpoints reanudables.

## Fontes activadas por defecto

### Castelán
- `wikimedia/wikipedia`, configuración `20231101.es`.
- `uonlp/CulturaX`, configuración `es` (pode requirir aceptar as condicións de acceso en Hugging Face).

Non se descargan por defecto `mC4` nin `OSCAR`: CulturaX xa se constrúe a partir deles e engadilos de novo xeraría moito solapamento.

`PleIAs/Spanish-PD-Books` é unha fonte valiosa de dominio público, pero queda en modo experimental porque o repositorio actual presenta inconsistencias de esquema entre shards.

### Código
- `common-pile/stackv2_edu_filtered`.

Esta variante parte de Stack V2 e conserva código de repositorios con licenzas abertas segundo os metadatos do dataset, ademais de aplicar filtrado educativo.

## Preparación en Windows

```powershell
cd "D:\CeltIA V4-2B"
.\.venv\Scripts\python.exe -m pip install -U datasets huggingface_hub
$env:HF_HOME="D:\corpus_llm_grande\_hf_cache"
$env:HF_DATASETS_CACHE="D:\corpus_llm_grande\_hf_cache\datasets"
$env:HF_HUB_CACHE="D:\corpus_llm_grande\_hf_cache\hub"
```

Se queres CulturaX, acepta previamente as condicións do dataset na túa conta de Hugging Face e inicia sesión:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

## Descarga recomendada

```powershell
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py
```

Límites por defecto:
- Wikipedia ES: ata 8 GB de JSONL resultante.
- CulturaX ES: ata 40 GB.
- StackV2 Edu: ata 60 GB.

Podes cambiar o límite por fonte:

```powershell
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py --max-gb 100
```

Ou descargar só unha parte:

```powershell
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py --groups spanish
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py --groups code
```

Para ampliar con mC4 e OSCAR, sabendo que aumentará o solapamento:

```powershell
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py --include-fallback-web
```

Para probar Spanish-PD-Books:

```powershell
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py --include-experimental --sources spanish_pd_books
```

## Filtros aplicados

Texto castelán:
- mínimo de lonxitude;
- proporción suficiente de texto alfabético;
- sinais léxicas frecuentes de castelán;
- control de caracteres non imprimibles;
- rexeitamento de repeticións masivas;
- rexeitamento de documentos cheos de URLs.

Código:
- exclúe vendor/generated cando o dataset o marca;
- exclúe rutas típicas `vendor`, `dist`, `build`, `node_modules`, minificados e sourcemaps;
- rexeita repetición extrema, liñas anormalmente longas e ficheiros dominados por símbolos;
- tenta inferir a linguaxe a partir dos metadatos ou extensión.

A deduplicación global exacta usa BLAKE2 + SQLite. Isto elimina duplicados idénticos, pero non substitúe unha etapa posterior de near-dedup semántica/minhash.

## Reanudación

Cada fonte mantén o seu estado en:

```text
D:\corpus_llm_grande\_state
```

Se se interrompe a descarga, repite o mesmo comando. O script retomará desde o número de filas xa inspeccionadas.

## Saída

```text
D:\corpus_llm_grande
├── castellano
│   ├── wikipedia_es.jsonl
│   └── culturax_es.jsonl
├── codigo
│   └── stackv2_edu.jsonl
├── _hf_cache
├── _state
├── manifest.json
└── run_summary.json
```

Cada liña JSONL conserva polo menos `text`, `source` e `dataset`; para código intenta conservar tamén `language`, `license`, `path` e `url`.

## Licenzas

Non asumas que “dataset público” significa “uso sen condicións”. Revisa e conserva os metadatos/licenzas antes de redistribuír o corpus ou un derivado. O descargador prioriza fontes con procedencia clara, pero a responsabilidade final de uso e redistribución segue dependendo da licenza de cada fonte/documento.


## Auditoría antes de entrenar

Despois dunha descarga parcial ou completa, executa:

```powershell
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py
```

O auditor percorre os JSONL en streaming e crea:

```text
D:\corpus_llm_grande\audit_report.json
```

Comproba por ficheiro:
- filas JSON mal formadas;
- documentos baleiros;
- duplicados exactos nunha mostra uniforme;
- repetición anormal de liñas;
- sinal de castelán >=95% para o corpus textual;
- rutas sospeitosas de vendor/build/minificado para código;
- distribución de linguaxes de programación;
- cobertura de metadatos de licenza;
- percentís de lonxitude de documento.

O comando devolve código 0 só se todos os ficheiros pasan os checks. Isto permite usalo como gate antes de tokenizar/adestrar.

Para unha mostra máis grande:

```powershell
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py --sample-size 50000
```

### Prensa histórica experimental

Tamén está dispoñible `PleIAs/Spanish-PD-Newspapers`, pero non se activa por defecto porque o OCR histórico pode introducir ruído. Para descargar libros e prensa de dominio público e sometelos despois á auditoría:

```powershell
.\.venv\Scripts\python.exe scripts\download_curated_corpus.py --include-experimental
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py --sample-size 50000
```

Non mestures automaticamente unha fonte experimental co corpus final se o seu informe non pasa. Se falla só por OCR/repetición, é mellor aplicar unha limpeza específica e volver auditar que relaxar o gate global.
