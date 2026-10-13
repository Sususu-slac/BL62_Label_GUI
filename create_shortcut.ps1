$ErrorActionPreference = "Stop"
$folder = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $desktop "BL62 Label GUI.lnk"))
$shortcut.TargetPath = Join-Path $env:SystemRoot "System32\wscript.exe"
$shortcut.Arguments = '"' + (Join-Path $folder "BL62 Label GUI.vbs") + '"'
$shortcut.WorkingDirectory = $folder
$icon = Join-Path $folder "BL62_Label.ico"
if (Test-Path $icon) { $shortcut.IconLocation = $icon }
$shortcut.Save()
Write-Output "Desktop shortcut created: BL62 Label GUI"
