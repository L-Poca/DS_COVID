# Script de lancement Streamlit DS_COVID - Version Simple
# =====================================================

Write-Host "Lancement Streamlit DS_COVID..." -ForegroundColor Green

# Se placer dans le répertoire src/streamlit (où se trouve ce script)
$currentLocation = Get-Location
Write-Host "Repertoire actuel: $currentLocation" -ForegroundColor Cyan

# Si on n'est pas dans src/streamlit, essayer de s'y rendre
if (-not (Test-Path "app.py")) {
    $streamlitDir = Join-Path $currentLocation "src\streamlit"
    if (Test-Path $streamlitDir) {
        Set-Location $streamlitDir
        Write-Host "Changement vers: $streamlitDir" -ForegroundColor Cyan
    }
}

# Vérifier que app.py existe maintenant
if (-not (Test-Path "app.py")) {
    Write-Host "ERREUR: app.py non trouve" -ForegroundColor Red
    Write-Host "Repertoire actuel: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Fichiers disponibles:" -ForegroundColor Yellow
    Get-ChildItem -Name "*.py" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  - $_" }
    Read-Host "Appuyez sur Entree"
    exit 1
}

Write-Host "OK: app.py trouve" -ForegroundColor Green

# Trouver l'environnement virtuel
Write-Host "Recherche de l'environnement virtuel..." -ForegroundColor Cyan

$venvFound = $false
$venvPath = ""
$activateScript = ""

# Essayer plusieurs chemins possibles
$possiblePaths = @(
    "..\..\\.venv\Scripts\activate.bat",
    "..\\.venv\Scripts\activate.bat", 
    ".venv\Scripts\activate.bat",
    "../../.venv/Scripts/activate.bat",
    "../.venv/Scripts/activate.bat",
    ".venv/Scripts/activate.bat"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $activateScript = $path
        $venvPath = Split-Path (Split-Path $path -Parent) -Parent
        $venvFound = $true
        Write-Host "OK: Environnement virtuel trouve via $path" -ForegroundColor Green
        break
    }
}

if (-not $venvFound) {
    Write-Host "ERREUR: Environnement virtuel non trouve" -ForegroundColor Red
    Write-Host "Chemins testes:" -ForegroundColor Yellow
    foreach ($path in $possiblePaths) {
        Write-Host "  - $path" -ForegroundColor Gray
    }
    Write-Host "Creez avec: python -m venv .venv" -ForegroundColor Yellow
    Read-Host "Appuyez sur Entree"
    exit 1
}

# Lancer Streamlit avec l'environnement virtuel
Write-Host "Activation et lancement de Streamlit..." -ForegroundColor Yellow
Write-Host "URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Utilisez Ctrl+C pour arreter" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

try {
    # Utiliser cmd pour éviter les problèmes PowerShell
    $command = "`"$activateScript`" && streamlit run app.py"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $command -Wait -NoNewWindow
    
    Write-Host "Termine." -ForegroundColor Green
} catch {
    Write-Host "ERREUR lors du lancement: $($_.Exception.Message)" -ForegroundColor Red
}

Read-Host "Appuyez sur Entree"
