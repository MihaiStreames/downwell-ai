[CmdletBinding(SupportsShouldProcess)]
param(
    [switch] $All
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Remove-Directory {
    [CmdletBinding(SupportsShouldProcess)]
    param([System.IO.DirectoryInfo] $Directory)

    $relativePath = Resolve-Path -Relative $Directory.FullName
    if ($PSCmdlet.ShouldProcess($relativePath, "remove")) {
        Write-Output $relativePath
        Remove-Item -Recurse -Force $Directory.FullName
    }
}

$cacheNames = "__pycache__", ".pytest_cache", ".ruff_cache"
Get-ChildItem -Directory -Recurse -Force |
    Where-Object { $cacheNames -contains $_.Name -and $_.FullName -notlike "*\.venv\*" } |
    ForEach-Object { Remove-Directory $_ }

if ($All) {
    foreach ($name in "runs", ".venv") {
        if (Test-Path $name) { Remove-Directory (Get-Item $name) }
    }

    Get-ChildItem -Directory -Recurse -Force -Filter "*.egg-info" |
        Where-Object { $_.FullName -notlike "*\.venv\*" } |
        ForEach-Object { Remove-Directory $_ }
}

Write-Output "clean complete"
