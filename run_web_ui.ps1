param(
    [switch]$Cuda,
    [int]$Port = 7860
)

$extra = if ($Cuda) { "cuda" } else { "cpu" }
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot ".uv-cache"
$env:PORT = "$Port"
uv run --extra $extra app.py
