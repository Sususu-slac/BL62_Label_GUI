Option Explicit
Dim shell, fso, folder, script, python, candidates, candidate, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
script = folder & "\app_en.py"
candidates = Array( _
  shell.ExpandEnvironmentStrings("%LOCALAPPDATA%\miniforge3\envs\BL62_Label_GUI\pythonw.exe"), _
  shell.ExpandEnvironmentStrings("%USERPROFILE%\miniforge3\envs\BL62_Label_GUI\pythonw.exe"), _
  shell.ExpandEnvironmentStrings("%LOCALAPPDATA%\miniforge3\envs\SAM3p1\pythonw.exe"), _
  shell.ExpandEnvironmentStrings("%USERPROFILE%\miniforge3\envs\SAM3p1\pythonw.exe") _
)
python = ""
For Each candidate In candidates
  If fso.FileExists(candidate) Then
    python = candidate
    Exit For
  End If
Next
If python = "" Then
  MsgBox "Python environment not found. Run install_windows.bat first.", 16, "BL62 Label GUI"
  WScript.Quit 1
End If
shell.CurrentDirectory = folder
command = Chr(34) & python & Chr(34) & " " & Chr(34) & script & Chr(34)
shell.Run command, 1, False
