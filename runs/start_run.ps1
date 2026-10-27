$ws = 'C:\Users\abc\Documents\Default Project'
$log = Join-Path $ws 'runs\run.log'
$err = Join-Path $ws 'runs\run.err'
New-Item -ItemType Directory -Force -Path (Join-Path $ws 'runs') | Out-Null

$py = Join-Path $ws '.venv\Scripts\python.exe'
# The workspace path contains a space, so every path argument must be quoted
# explicitly -- Start-Process -ArgumentList does not quote for us.
$q = { param($p) '"' + $p + '"' }
$args = @(
    '-m', 'cgate.experiments.run',
    '--data', 'brats',
    '--root', (& $q (Join-Path $ws 'data')),
    '--experts', 'torch',
    '--pretrained',
    '--arch', 'resnet50',
    '--epochs', '12',
    '--batch-size', '32',
    '--table-cache', (& $q (Join-Path $ws 'runs\table')),
    '--slice-cache', (& $q (Join-Path $ws 'runs\cache')),
    '--out', (& $q (Join-Path $ws 'runs\brats_full_seed0.json')),
    '--figures', (& $q (Join-Path $ws 'runs\figs')),
    '--predictions', (& $q (Join-Path $ws 'runs\predictions.json')),
    '--seed', '0'
)

Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $ws `
    -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow
Write-Output "launched detached run; logs at $log"
