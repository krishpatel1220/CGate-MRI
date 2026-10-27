# Package the shareable results zip: source + results + reports + docs.
# Usage:
#   powershell -File runs\make_zip.ps1          -> cgate-mri-results.zip  (light, no model)
#   powershell -File runs\make_zip.ps1 -Full    -> cgate-mri-full-demo.zip (runnable bundle)
param([switch]$Full)
$ErrorActionPreference = 'Stop'
$ws  = 'C:\Users\abc\Documents\Default Project'
$stg = Join-Path $ws 'runs\_zip_staging'
$zip = Join-Path $ws $(if ($Full) { 'cgate-mri-full-demo.zip' } else { 'cgate-mri-results.zip' })

if (Test-Path $stg) { Remove-Item $stg -Recurse -Force }
if (Test-Path $zip) { Remove-Item $zip -Force }
New-Item -ItemType Directory -Force -Path $stg | Out-Null

# 1. Source (reporting module is the new deliverable)
$src = Join-Path $stg 'source'
Copy-Item "$ws\cgate-mri-main\src"  "$src\src"  -Recurse
Copy-Item "$ws\cgate-mri-main\tests" "$src\tests" -Recurse
Copy-Item "$ws\cgate-mri-main\docs"  "$src\docs"  -Recurse
Copy-Item "$ws\cgate-mri-main\pyproject.toml","$ws\cgate-mri-main\README.md" $src
Get-ChildItem $src -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
Get-ChildItem $src -Recurse -Directory -Filter '.pytest_cache' | Remove-Item -Recurse -Force

# 1b. Flask web demo (source only; runs\system_B6.pkl is excluded - regenerate
#     with tools\save_system.py, see web\README.md)
$web = Join-Path $stg 'web'
Copy-Item "$ws\web" $web -Recurse
Get-ChildItem $web -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
if (Test-Path "$ws\runs\test_uploads\BraTS2021_00009_demo.zip") {
    # half-res sample upload (1.5 MB) so the demo can be tried immediately
    Copy-Item "$ws\runs\test_uploads\BraTS2021_00009_demo.zip" "$web\sample_upload.zip"
}


# 2. Results
$res = Join-Path $stg 'results'
New-Item -ItemType Directory -Force -Path $res | Out-Null
Copy-Item "$ws\runs\brats_full_seed0.json" $res
Copy-Item "$ws\runs\README_RESULTS.txt" $res
if (Test-Path "$ws\runs\figs") { Copy-Item "$ws\runs\figs" "$res\figs" -Recurse }

# 3. Predictions export (inputs for cgate-report)
if (Test-Path "$ws\runs\predictions.json") { Copy-Item "$ws\runs\predictions.json" $res }

# 4. Reports: a sample of template reports + all VLM ones + index files
$rp = Join-Path $stg 'reports'
New-Item -ItemType Directory -Force -Path "$rp\template_sample","$rp\vlm" | Out-Null
if (Test-Path "$ws\runs\reports\index.md") {
    Copy-Item "$ws\runs\reports\index.md" "$rp\template_sample\index.md"
    Get-ChildItem "$ws\runs\reports" -Filter 'BraTS*.md' |
        Select-Object -First 40 | ForEach-Object { Copy-Item $_.FullName "$rp\template_sample\" }
    # include the deferred ones too (they demonstrate the gate notice)
    $idx = Get-Content "$ws\runs\reports\index.md"
    foreach ($line in ($idx | Where-Object { $_ -match 'deferred' })) {
        if ($line -match '\(([^)]+)\.md\)') {
            Copy-Item "$ws\runs\reports\$($Matches[1]).md" "$rp\template_sample\" -ErrorAction SilentlyContinue
        }
    }
}
if (Test-Path "$ws\runs\reports_vlm") {
    Copy-Item "$ws\runs\reports_vlm\*" "$rp\vlm\" -Recurse -Force
}

# 5. Monitor + launcher + model-save scripts
$tools = Join-Path $stg 'tools'
New-Item -ItemType Directory -Force -Path $tools | Out-Null
Copy-Item "$ws\runs\monitor.ps1","$ws\runs\start_run.ps1" $tools
if (Test-Path "$ws\tools\save_system.py") { Copy-Item "$ws\tools\save_system.py" $tools }

# 6. Start-here guide (lands at the zip root)
Copy-Item "$ws\START_HERE.md" $stg

# 7. Full bundle only: the fitted model + everything web\test_demo.py needs,
#    so an unzipped folder runs and its tests pass without the BraTS dataset.
if ($Full) {
    $r = Join-Path $stg 'runs'
    New-Item -ItemType Directory -Force -Path "$r\cache","$r\test_uploads" | Out-Null
    Copy-Item "$ws\runs\system_B6.pkl" $r
    Copy-Item "$ws\runs\predictions.json" $r
    Copy-Item "$ws\runs\cache\index.json" "$r\cache\"
    Get-ChildItem "$ws\runs\test_uploads" -File | Copy-Item -Destination "$r\test_uploads\"
}

# 8. Compress (.NET: Compress-Archive is slow/memory-hungry on the 174 MB pkl)
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path $zip) { Remove-Item $zip -Force }
[IO.Compression.ZipFile]::CreateFromDirectory($stg, $zip, [IO.Compression.CompressionLevel]::Optimal, $false)
Remove-Item $stg -Recurse -Force

$kb = [math]::Round((Get-Item $zip).Length/1KB, 0)
Write-Output "Wrote $zip  ($kb KB)"
