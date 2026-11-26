; Bộ cài Windows cho Beauty HDR (Inno Setup 6). build_windows.ps1 gọi file này, hoặc chạy tay:
;   ISCC /DAppVersion=0.10.1 /DFlavor=cpu /DWithModels=1 packaging\NetAnh.iss
; Đọc bản đóng gói PyInstaller ở DistDir (mặc định _build\dist\NetAnh), ra _build\output\NetAnh-<phiên bản>-setup.exe.
; Giữ file này ở dạng UTF-8 có BOM để chữ tiếng Việt trong bộ cài hiện đúng.
; Cài cho riêng người dùng (không cần quyền quản trị) vào %LOCALAPPDATA%\Programs\NetAnh, nên app ghi được
; thư mục weights cạnh NetAnh.exe để tải thêm model.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Flavor
  #define Flavor "cpu"
#endif
#ifndef WithModels
  #define WithModels "1"
#endif
#ifndef DistDir
  #define DistDir "..\_build\dist\NetAnh"
#endif

[Setup]
AppId={{6B0F3E2A-8C1D-4F7E-9A51-2D7C3B9E4A10}
AppName=Beauty HDR
AppVersion={#AppVersion}
AppVerName=Beauty HDR {#AppVersion} ({#Flavor})
AppPublisher=Beauty HDR
DefaultDirName={localappdata}\Programs\NetAnh
DefaultGroupName=Beauty HDR
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\_build\output
#if Flavor == "cuda"
OutputBaseFilename=NetAnh-{#AppVersion}-GPU-NVIDIA-setup
#elif Flavor == "cuda_old"
OutputBaseFilename=NetAnh-{#AppVersion}-GPU-NVIDIA-doi-cu-setup
#else
OutputBaseFilename=NetAnh-{#AppVersion}-setup
#endif
SetupIconFile=net_anh.ico
UninstallDisplayIcon={app}\NetAnh.exe
Compression=lzma2/normal
SolidCompression=no
; bản CUDA vượt 2 GB: tách thành setup.exe + các file .bin (để cùng thư mục khi cài). Bản CPU để một file
; setup.exe duy nhất: DiskSpanning=yes luôn tách dữ liệu ra .bin, người tải mỗi setup.exe sẽ cài lỗi.
#if Flavor == "cuda" || Flavor == "cuda_old"
DiskSpanning=yes
DiskSliceSize=1900000000
#endif
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Tạo biểu tượng ngoài màn hình (Desktop)"; Flags: unchecked

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Excludes: "weights\*"; Flags: ignoreversion recursesubdirs createallsubdirs
#if WithModels == "1"
Source: "{#DistDir}\weights\*.pth"; DestDir: "{app}\weights"; Flags: ignoreversion
Source: "{#DistDir}\weights\*.onnx"; DestDir: "{app}\weights"; Flags: ignoreversion skipifsourcedoesntexist
#endif

[Dirs]
; users-modify: cả khi chọn cài cho mọi người dùng (Program Files), app vẫn tải thêm model được (RestoreFormer++, DAT-2)
Name: "{app}\weights"; Permissions: users-modify

[Icons]
Name: "{group}\Beauty HDR"; Filename: "{app}\NetAnh.exe"
Name: "{group}\Beauty HDR - Máy chủ cho điện thoại"; Filename: "{app}\NetAnh.exe"; Parameters: "--may-chu"
Name: "{group}\Gỡ Beauty HDR"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Beauty HDR"; Filename: "{app}\NetAnh.exe"; Tasks: desktopicon

[Run]
; máy chủ cho điện thoại: mở cổng 8765 (HTTP) và 8766 (UDP tìm máy) cho mạng Riêng tư, không mở cho mạng công cộng
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Beauty HDR máy chủ"" dir=in action=allow protocol=TCP localport=8765 profile=private"; Flags: runhidden; Check: IsAdminInstallMode
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Beauty HDR tìm máy"" dir=in action=allow protocol=UDP localport=8766 profile=private"; Flags: runhidden; Check: IsAdminInstallMode
Filename: "{app}\NetAnh.exe"; Description: "Mở Beauty HDR"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; model tải thêm sau khi cài và file log tự kiểm
Type: filesandordirs; Name: "{app}\weights"
Type: files; Name: "{app}\tu_kiem.log"
