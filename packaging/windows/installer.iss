; Inno Setup script: builds ShrinkageDenoiser-Setup-<version>.exe from the PyInstaller output in dist\ShrinkageDenoiser
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
[Setup]
AppId={{6F1B7B52-3C1D-4D6B-9B53-5A0C2D7E9A11}
AppName=Shrinkage Denoiser
AppVersion={#AppVersion}
AppPublisher=Shrinkage Denoiser
DefaultDirName={autopf}\Shrinkage Denoiser
DefaultGroupName=Shrinkage Denoiser
OutputDir=..\..\installer
OutputBaseFilename=ShrinkageDenoiser-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayName=Shrinkage Denoiser
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\..\dist\ShrinkageDenoiser\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Shrinkage Denoiser"; Filename: "{app}\ShrinkageDenoiser.exe"
Name: "{autodesktop}\Shrinkage Denoiser"; Filename: "{app}\ShrinkageDenoiser.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ShrinkageDenoiser.exe"; Description: "Launch Shrinkage Denoiser"; Flags: nowait postinstall skipifsilent
