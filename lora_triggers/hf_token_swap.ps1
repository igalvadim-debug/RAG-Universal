$ErrorActionPreference = 'SilentlyContinue'
$base = 'D:\claudeAgent\RAG\workflow\lora_triggers'
$log  = Join-Path $base 'swap.log'
'' | Set-Content $log -Encoding UTF8
function L($m){ Add-Content -Path $log -Value $m -Encoding UTF8 }

$envFile = 'D:\claudeAgent\Coder\setting_json\.env'
$lines = Get-Content $envFile -Encoding UTF8
$new = [regex]::Match(($lines | Where-Object { $_ -match 'read und write' } | Select-Object -First 1), 'hf_[A-Za-z0-9]{30,}').Value
$old = [regex]::Match(($lines | Where-Object { $_ -match '^\s*setx HF_TOKEN' } | Select-Object -First 1), 'hf_[A-Za-z0-9]{30,}').Value
if (-not $new -or -not $old -or $new -eq $old) { L 'ABBRUCH: Tokens nicht eindeutig ermittelt'; L 'FERTIG'; return }
L ("neuer Token ...{0}, alter Token ...{1}" -f $new.Substring($new.Length-4), $old.Substring($old.Length-4))

# 1) Windows-Benutzervariable
$cur = [Environment]::GetEnvironmentVariable('HF_TOKEN','User')
if ($cur -eq $old) { [Environment]::SetEnvironmentVariable('HF_TOKEN', $new, 'User'); L 'USERENV: HF_TOKEN auf neuen Token gesetzt' }
elseif ($cur -eq $new) { L 'USERENV: HF_TOKEN war schon neu' }
else { L ('USERENV: HF_TOKEN hat anderen Wert (Ende ...{0}) - nicht geaendert' -f $(if($cur){$cur.Substring([Math]::Max(0,$cur.Length-4))}else{'leer'})) }
foreach($v in 'HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_API_KEY','HF_API_KEY'){
  foreach($scope in 'User','Machine'){ $x=[Environment]::GetEnvironmentVariable($v,$scope); if($x){ L ("ENV-HINWEIS: {0} ({1}) gesetzt, Ende ...{2}" -f $v,$scope,$x.Substring([Math]::Max(0,$x.Length-4))) } }
}
$m = [Environment]::GetEnvironmentVariable('HF_TOKEN','Machine'); if($m){ L ("ENV-HINWEIS: HF_TOKEN (Machine) gesetzt, Ende ...{0}" -f $m.Substring([Math]::Max(0,$m.Length-4))) }

# 2) setx-Zeile in der Sammel-.env
$raw = [System.IO.File]::ReadAllText($envFile)
$setxOld = 'setx HF_TOKEN "' + $old + '"'
if ($raw.Contains($setxOld)) {
  $b = $envFile + '.bak_20261010'; if (-not (Test-Path $b)) { Copy-Item $envFile $b }
  [System.IO.File]::WriteAllText($envFile, $raw.Replace($setxOld, 'setx HF_TOKEN "' + $new + '"'), (New-Object System.Text.UTF8Encoding $false))
  L 'SAMMEL-ENV: setx-Zeile aktualisiert (Notizzeilen unveraendert)'
}

# 3) Dateien durchsuchen
$inc  = '*.env','.env','.env.*','*.bat','*.cmd','*.ps1','*.json','*.yaml','*.yml','*.py','*.txt','*.toml','*.ini','*.cfg','*.conf','*.sh','*.js','*.ts'
$skip = '\\(node_modules|venv|\.venv|env|\.git|site-packages|__pycache__|\$Recycle\.Bin|Windows|Program Files|Program Files \(x86\)|AppData\\Local\\(Temp|Microsoft|Packages|pip|npm-cache)|AppData\\Roaming\\(npm|Code|Microsoft)|models|output|input|\.cache\\huggingface\\hub)\\'
$roots = 'D:\','C:\Users\Startklar'
$scanned = 0; $changed = 0
foreach ($root in $roots) {
  Get-ChildItem -Path $root -Recurse -Force -File -Include $inc -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -lt 5MB -and $_.FullName -notmatch $skip -and $_.Name -notmatch 'backup|\.bak' -and $_.FullName -ne $envFile -and $_.Name -ne 'hf_token_swap.ps1' } |
    ForEach-Object {
      $scanned++
      $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
      $head = $bytes[0..([Math]::Min(4095,$bytes.Length-1))]
      if ($head -contains 0) { return }
      $bom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
      $txt = [System.Text.Encoding]::UTF8.GetString($bytes)
      if ($bom) { $txt = $txt.Substring(1) }
      $found = [regex]::Matches($txt, 'hf_[A-Za-z0-9]{30,}') | ForEach-Object { $_.Value } | Select-Object -Unique
      if (-not $found) { return }
      if ($found -contains $old) {
        $b = $_.FullName + '.bak_20261010'; if (-not (Test-Path $b)) { Copy-Item $_.FullName $b }
        $out = $txt.Replace($old, $new)
        $enc = New-Object System.Text.UTF8Encoding $bom
        [System.IO.File]::WriteAllText($_.FullName, $out, $enc)
        $changed++
        L ("GEAENDERT: {0}" -f $_.FullName)
      }
      foreach ($f in $found) { if ($f -ne $old -and $f -ne $new) { L ("ANDERER-TOKEN (nicht angefasst): {0}  Ende ...{1}" -f $_.FullName, $f.Substring($f.Length-4)) } elseif ($f -eq $new -and -not ($found -contains $old)) { L ("SCHON-NEU: {0}" -f $_.FullName) } }
    }
}
L ("Dateien geprueft: {0}, geaendert: {1}" -f $scanned, $changed)
L 'FERTIG'
