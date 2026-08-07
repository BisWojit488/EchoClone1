param(
    [switch]$Cuda
)

$extra = if ($Cuda) { "cuda" } else { "cpu" }
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot ".uv-cache"
uv run --extra $extra demo_toolbox.py
