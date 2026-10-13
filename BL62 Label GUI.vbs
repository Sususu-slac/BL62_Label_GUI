Option Explicit
Dim shell, folder, command
Set shell = CreateObject("WScript.Shell")
folder = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
command = "cmd.exe /c """ & folder & "\run_gui.bat"""
shell.CurrentDirectory = folder
shell.Run command, 0, False
