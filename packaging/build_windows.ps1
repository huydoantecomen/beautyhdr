# Tạo file cài đặt Windows cho Beauty HDR. Bấm đúp TAO_FILE_CAI_DAT.bat (bản gửi người dùng) hoặc dong_goi_exe.bat
# (trong thư mục mã nguồn) là xong: ra MỘT file NetAnh-<phiên bản>-setup.exe, Explorer tự mở và chọn sẵn file đó.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1                 # bản CPU, kèm model, ra bộ cài
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Flavor cuda    # bản GPU NVIDIA (trên 2 GB: thêm file .bin)
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -NoModels -NoInstaller -OutDir D:\ra
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -CapNhat      # CAP_NHAT.bat: cập nhật bản đã cài
#
# -CapNhat (gói BeautyHDR-<phiên bản>-CapNhat.zip): dùng lại môi trường build cũ trong _build (không tải lại torch),
# không tải model, không tạo bộ cài; đóng gói xong thì chép đè lên thư mục Beauty HDR đã cài (giữ nguyên thư mục
# weights và bộ gỡ cài đặt) rồi chạy tự kiểm bản đã cài với model sẵn có.
#
# Các bước: tìm Python 3.10-3.13 (không có thì tự cài 3.12) -> môi trường build riêng -> PyInstaller -> tải model
# -> tự kiểm bản đóng gói -> Inno Setup (không có thì tự cài) -> chép file cài đặt ra OutDir (mặc định: thư mục mã
# nguồn). Mọi file trung gian nằm trong _build\, xoá được sau khi xong; chạy lại thì dùng lại phần đã tải.
# Chạy được trong đường dẫn có dấu tiếng Việt và dấu cách. Giữ file này ở dạng UTF-8 CÓ BOM: PowerShell 5.1 đọc
# file không BOM theo bảng mã ANSI, chữ có dấu thành ký tự lạ và có thể làm hỏng cú pháp.
param(
    [ValidateSet("cpu", "cuda", "cuda_old", "xpu")] [string]$Flavor = "cpu",  # cuda_old: GTX 10 / 900 (CUDA 12.6)
    [switch]$NoModels,
    [switch]$NoInstaller,
    [string]$Python = "",  # ví dụ "py -3.12" hoặc "python"; để trống thì tự tìm
    [string]$OutDir = "",
    [switch]$CapNhat
)
if ($CapNhat) { $NoModels = $true; $NoInstaller = $true }
$FlavorGiven = $PSBoundParameters.ContainsKey("Flavor")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # PowerShell 5.1 tải file chậm hẳn khi vẽ thanh tiến độ
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }  # đọc đúng chữ có dấu Python in ra
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

# Bản đầy đủ tự phát hiện card NVIDIA: bản CPU chạy chậm gấp ~10 lần trên máy có card (lỗi 0.12.0 trên GTX 1660 Ti)
if (-not $CapNhat -and -not $FlavorGiven -and $env:CI -ne "true") {
    $nv = $null
    try { $nv = Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1 } catch { }
    if ($nv) {
        Write-Host "Máy có card $($nv.Name). Bản GPU NVIDIA nhanh hơn khoảng 10 lần nhưng nặng hơn (khoảng 3 GB, bộ cài chia thành vài file)." -ForegroundColor Yellow
        $ans = Read-Host "Tạo bản GPU NVIDIA? Gõ Enter để chọn GPU, hoặc gõ C rồi Enter để chọn bản CPU"
        # GTX 10 / 900, MX1xx–3xx (Pascal, Maxwell): PyTorch CUDA 12.8 trở lên bỏ các dòng này, dùng bản CUDA 12.6
        $old = $nv.Name -match "GTX\s*(10[5-8]0|9[5-8]0|7[5-9]0)|MX\s*[1-3][1-5]0|GT\s*10[3-9]0|TITAN X(p)?\b"
        $Flavor = if ($ans -match "^[cC]") { "cpu" } elseif ($old) { "cuda_old" } else { "cuda" }
    }
}

# Chọn bản GPU NVIDIA (kể cả CHUYEN_SANG_GPU.bat truyền -Flavor cuda) trên máy chỉ có card GTX 10 / 900: bản CUDA 12.8 không
# chạy trên các dòng này, tự đổi sang bản cho card đời cũ (CUDA 12.6).
if ($Flavor -eq "cuda" -and $env:CI -ne "true") {
    $nvs = @()
    try { $nvs = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match "NVIDIA" } | ForEach-Object { $_.Name }) } catch { }
    $legacy = "GTX\s*(10[5-8]0|9[5-8]0|7[5-9]0)|MX\s*[1-3][1-5]0|GT\s*10[3-9]0|TITAN X(p)?\b"
    if ($nvs.Count -gt 0 -and @($nvs | Where-Object { $_ -notmatch $legacy }).Count -eq 0) {
        Write-Host "Card $($nvs[0]) là đời cũ (Pascal/Maxwell): dùng bản GPU cho card đời cũ (PyTorch CUDA 12.6)." -ForegroundColor Yellow
        $Flavor = "cuda_old"
    }
}

$Root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = $Root }
$OutDir = [IO.Path]::GetFullPath($OutDir)
Set-Location -LiteralPath $Root
$Work = Join-Path $Root "_build"
$Venv = Join-Path $Work "venv-$Flavor"
$DistRoot = Join-Path $Work "dist"
$Dist = Join-Path $DistRoot "NetAnh"
$Output = Join-Path $Work "output"
$Weights = Join-Path $Work "weights"

function Step([string]$msg) {
    Write-Host ""
    Write-Host "== $msg" -ForegroundColor Yellow
}

# Chạy chương trình ngoài, dừng nếu mã thoát khác 0. Tham số có dấu cách được PowerShell tự đặt trong ngoặc kép.
# Xét lỗi theo mã thoát, không theo stderr: pip và PyInstaller ghi cả thông báo thường ra stderr.
function Exec([string]$exe, [string[]]$argv) {
    Write-Host ">> $exe $($argv -join ' ')" -ForegroundColor Cyan
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $exe @argv
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) { throw "Lệnh lỗi (mã $code): $exe $($argv -join ' ')" }
}

# Chạy chương trình ngoài, bỏ qua lỗi (winget báo mã khác 0 cả khi đã cài sẵn). Chữ chương trình in ra đi thẳng ra
# màn hình (Out-Host): trong hàm PowerShell, output không bắt lại sẽ thành giá trị trả về của hàm (lỗi 0.7.0: chữ của
# winget lẫn vào đường dẫn python.exe / ISCC.exe, báo "is not recognized as the name of a cmdlet").
function Try-Exec([string]$exe, [string[]]$argv) {
    Write-Host ">> $exe $($argv -join ' ')" -ForegroundColor Cyan
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @argv | Out-Host } catch { Write-Host $_.Exception.Message }
    $ErrorActionPreference = $old
}

function Download([string]$url, [string]$dest) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Write-Host ">> tải $url" -ForegroundColor Cyan
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $dest
}

function Has-Command([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# Python 3.10-3.13 có sẵn: trả về mảng (lệnh, tham số...) hoặc $null.
# "python" có thể chỉ là lối tắt mở Microsoft Store (in lỗi, mã 9009): kiểm bằng cách chạy thật.
function Find-Python {
    $cands = @()
    if ($Python) {
        $cands += , @($Python -split "\s+")
    } else {
        $cands += , @("py", "-3.12")
        $cands += , @("py", "-3.11")
        $cands += , @("py", "-3.13")
        $cands += , @("python")
        $cands += , @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")
    }
    foreach ($c in $cands) {
        $exe = $c[0]
        $rest = @($c | Select-Object -Skip 1)
        if (-not (Has-Command $exe)) { continue }
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $v = ""
        try { $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null } catch { $v = "" }
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
        if ($code -eq 0 -and "$v".Trim() -match '^3\.1[0-3]$') {
            Write-Host "Dùng Python $("$v".Trim()): $($c -join ' ')"
            return , $c
        }
    }
    return $null
}

function Install-Python {
    Step "Máy chưa có Python 3.10-3.13: tự cài Python 3.12 cho riêng tài khoản này"
    $p = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    if (Has-Command "winget") {
        Try-Exec "winget" @("install", "-e", "--id", "Python.Python.3.12", "--scope", "user", "--silent",
                            "--accept-package-agreements", "--accept-source-agreements")
    }
    if (-not (Test-Path -LiteralPath $p)) {
        try {
            $setup = Join-Path $Work "python-3.12.10-amd64.exe"
            Download "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" $setup
            $pr = Start-Process -FilePath $setup -ArgumentList "/quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_launcher=0" -Wait -PassThru
            Write-Host "Cài Python xong, mã $($pr.ExitCode)"
        } catch { Write-Host $_.Exception.Message }
    }
    if (-not (Test-Path -LiteralPath $p)) {
        throw "Không tự cài được Python. Cài Python 3.12 từ https://www.python.org/downloads/ (tick 'Add python.exe to PATH') rồi bấm lại file .bat."
    }
    return , @($p)
}

function Find-Iscc {
    $c = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
           "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")
    foreach ($p in $c) { if (Test-Path -LiteralPath $p) { return $p } }
    return $null
}

function Install-InnoSetup {
    Step "Chưa có Inno Setup 6 (công cụ làm file cài đặt, miễn phí): tự cài"
    if (Has-Command "winget") {
        Try-Exec "winget" @("install", "-e", "--id", "JRSoftware.InnoSetup", "--scope", "user", "--silent",
                            "--accept-package-agreements", "--accept-source-agreements")
        if (-not (Find-Iscc)) {
            Try-Exec "winget" @("install", "-e", "--id", "JRSoftware.InnoSetup", "--silent",
                                "--accept-package-agreements", "--accept-source-agreements")
        }
    }
    if (-not (Find-Iscc)) {
        try {
            $setup = Join-Path $Work "innosetup.exe"
            Download "https://jrsoftware.org/download.php/is.exe" $setup
            $pr = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /CURRENTUSER" -Wait -PassThru
            Write-Host "Cài Inno Setup xong, mã $($pr.ExitCode)"
        } catch { Write-Host $_.Exception.Message }
    }
    $iscc = Find-Iscc
    if (-not $iscc) {
        throw "Không tự cài được Inno Setup. Tải tại https://jrsoftware.org/isdl.php, cài xong bấm lại file .bat (phần đã làm không phải làm lại)."
    }
    return $iscc
}

# Thư mục Beauty HDR đã cài: đọc từ khoá gỡ cài đặt của Inno Setup (AppId trong NetAnh.iss), không có thì thư mục mặc định
function Find-Install {
    $sub = "Software\Microsoft\Windows\CurrentVersion\Uninstall\{6B0F3E2A-8C1D-4F7E-9A51-2D7C3B9E4A10}_is1"
    foreach ($path in @("HKCU:\$sub", "HKLM:\$sub", "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{6B0F3E2A-8C1D-4F7E-9A51-2D7C3B9E4A10}_is1")) {
        try {
            $loc = (Get-ItemProperty -LiteralPath $path -ErrorAction Stop).InstallLocation
            if ($loc -and (Test-Path -LiteralPath (Join-Path $loc "NetAnh.exe"))) { return @($loc.TrimEnd("\"), $path) }
        } catch { }
    }
    $def = Join-Path $env:LOCALAPPDATA "Programs\NetAnh"
    if (Test-Path -LiteralPath (Join-Path $def "NetAnh.exe")) { return @($def, "") }
    return $null
}

try {
    if ($CapNhat) {
        Step "0/6 Tìm Beauty HDR đã cài"
        $found = Find-Install
        if (-not $found) {
            throw "Chưa thấy Beauty HDR cài trên máy này. Hãy cài bản đầy đủ (file setup.exe) trước, rồi mới dùng gói Cập nhật."
        }
        $AppDir = $found[0]; $UninstKey = $found[1]
        Write-Host "Đã cài ở: $AppDir"
        $wdir = Join-Path $AppDir "weights"
        $nModel = @(Get-ChildItem -LiteralPath $wdir -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in ".pth", ".onnx" }).Count
        Write-Host "Model sẵn có: $nModel file trong $wdir (giữ nguyên, không tải lại)"
        if (-not $FlavorGiven) {
            $cuda = @(Get-ChildItem -LiteralPath (Join-Path $AppDir "_internal\torch\lib") -Filter "*cuda*.dll" -ErrorAction SilentlyContinue).Count -gt 0
            $xpuDll = @(Get-ChildItem -LiteralPath (Join-Path $AppDir "_internal\torch\lib") -Filter "*sycl*.dll" -ErrorAction SilentlyContinue).Count -gt 0
            $Flavor = if ($cuda) { "cuda" } elseif ($xpuDll) { "xpu" } else { "cpu" }
            $tv = Join-Path $AppDir "_internal\torch\version.py"
            if ($cuda -and (Test-Path -LiteralPath $tv) -and ((Get-Content -LiteralPath $tv -Raw) -match "cuda\s*(:\s*\w+\s*)?=\s*'12\.6")) { $Flavor = "cuda_old" }
            $Venv = Join-Path $Work "venv-$Flavor"
        }
        Write-Host "Bản đã cài: $Flavor"
        if (-not (Test-Path -LiteralPath (Join-Path $Venv "Scripts\python.exe"))) {
            Write-Host "Không thấy môi trường build cũ ($Venv): phải tải lại thư viện (torch), lâu hơn. Model vẫn KHÔNG tải lại." -ForegroundColor Yellow
            Write-Host "Muốn nhanh: đóng cửa sổ này, chép gói Cập nhật vào ĐÚNG thư mục chứa bản cũ (có thư mục ma-nguon\_build) rồi chạy lại." -ForegroundColor Yellow
        }
    }
    New-Item -ItemType Directory -Force -Path $Work | Out-Null
    $Version = (Select-String -LiteralPath (Join-Path $Root "net_anh\__init__.py") -Pattern '__version__ = "([^"]+)"').Matches[0].Groups[1].Value
    Write-Host "Tạo file cài đặt Beauty HDR $Version (bản $Flavor). Lần đầu mất khoảng 20-40 phút và cần mạng; đừng tắt cửa sổ này." -ForegroundColor Green

    Step "1/6 Python"
    $PyCmd = Find-Python
    if (-not $PyCmd) {
        # cài xong thì tìm lại theo đường dẫn chuẩn, không dùng giá trị trả về của hàm cài
        Install-Python | Out-Null
        $PyCmd = Find-Python
        if (-not $PyCmd) {
            throw "Đã cài Python nhưng chưa chạy được. Đóng cửa sổ này rồi bấm lại file .bat."
        }
    }
    $Py = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Py)) {
        Exec $PyCmd[0] (@($PyCmd | Select-Object -Skip 1) + @("-m", "venv", $Venv))
    }

    Step "2/6 Cài thư viện vào môi trường build riêng (_build\venv-$Flavor), không đụng tới Python đang dùng"
    $TorchIndex = switch ($Flavor) { "cuda" { "https://download.pytorch.org/whl/cu128" } "cuda_old" { "https://download.pytorch.org/whl/cu126" } "xpu" { "https://download.pytorch.org/whl/xpu" } default { "https://download.pytorch.org/whl/cpu" } }
    $keepTorch = $false
    if ($CapNhat) {
        & $Py (Join-Path $Root "packaging\kiem_torch.py")
        $keepTorch = ($LASTEXITCODE -eq 0)
        if ($keepTorch) { Write-Host "Dùng lại torch đã cài trong môi trường build cũ, không tải lại." -ForegroundColor Green }
    }
    if (-not $keepTorch) {
        Exec $Py @("-m", "pip", "install", "--upgrade", "pip")
        # --upgrade: luôn cài torch và torchvision cùng một đợt từ kho PyTorch để hai bản khớp nhau, kể cả khi dùng lại
        # môi trường build cũ (lỗi 0.8.1: "operator torchvision::nms does not exist")
        Exec $Py @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", $TorchIndex)
    }
    Exec $Py @("-m", "pip", "install", "-r", "requirements.txt", "pyinstaller")
    & $Py (Join-Path $Root "packaging\kiem_torch.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "torch/torchvision không khớp: cài lại cả hai (khoảng 1-3 GB)…" -ForegroundColor Yellow
        Exec $Py @("-m", "pip", "uninstall", "-y", "torch", "torchvision")
        Exec $Py @("-m", "pip", "install", "--no-cache-dir", "torch", "torchvision", "--index-url", $TorchIndex)
        Exec $Py @((Join-Path $Root "packaging\kiem_torch.py"))
    }

    Step "3/6 Đóng gói bằng PyInstaller"
    Exec $Py @("-m", "PyInstaller", "--noconfirm", "--clean", "--workpath", (Join-Path $Work "pyinstaller-$Flavor"),
               "--distpath", $DistRoot, "packaging\net_anh.spec")
    # bộ preset mẫu (ảnh thẻ 4 nhóm) nằm cạnh NetAnh.exe: Quản lý… > Nạp từ file… > thư mục presets
    $presetSrc = Join-Path $Root "presets"
    if (Test-Path -LiteralPath $presetSrc) {
        Copy-Item -LiteralPath $presetSrc -Destination (Join-Path $Dist "presets") -Recurse -Force
    }

    if (-not $NoModels) {
        Step "4/6 Tải model AI (khoảng 620 MB, kiểm SHA-256; lần sau dùng lại bản đã tải)"
        Exec $Py @("app.py", "--tai-model", $Weights)
        $wd = Join-Path $Dist "weights"
        New-Item -ItemType Directory -Force -Path $wd | Out-Null
        Get-ChildItem -LiteralPath $Weights -Filter "*.pth" | Copy-Item -Destination $wd -Force
        Get-ChildItem -LiteralPath $Weights -Filter "*.onnx" | Copy-Item -Destination $wd -Force  # đoán tuổi, giới tính
    } else {
        Step "4/6 Bỏ qua model (app tự tải khi cần)"
    }

    Step "5/6 Tự kiểm bản đóng gói (chạy thử chế độ Nhanh và Chân dung AI)"
    $Log = Join-Path $Work "tu_kiem_$Flavor.log"
    if (Test-Path -LiteralPath $Log) { Remove-Item -LiteralPath $Log -Force }
    $tk = "--tu-kiem `"$Log`""
    if (-not $NoModels) { $tk += " --can-ai" }
    $pr = Start-Process -FilePath (Join-Path $Dist "NetAnh.exe") -ArgumentList $tk -Wait -PassThru
    if (Test-Path -LiteralPath $Log) { Get-Content -LiteralPath $Log -Encoding UTF8 | ForEach-Object { Write-Host $_ } }
    if ($pr.ExitCode -ne 0) { throw "Tự kiểm KHÔNG ĐẠT (mã $($pr.ExitCode)), xem $Log" }

    if ($CapNhat) {
        Step "6/6 Chép đè lên bản đã cài (giữ thư mục weights và bộ gỡ cài đặt)"
        if (Get-Process -Name "NetAnh" -ErrorAction SilentlyContinue) {
            Write-Host "Beauty HDR đang mở. Hãy đóng app (cả cửa sổ máy chủ nếu có) rồi bấm Enter để chép tiếp." -ForegroundColor Yellow
            Read-Host | Out-Null
            if (Get-Process -Name "NetAnh" -ErrorAction SilentlyContinue) { throw "Beauty HDR vẫn đang mở, chưa chép được. Đóng app rồi chạy lại CAP_NHAT.bat." }
        }
        # /MIR: bản cài giống hệt bản mới (xoá file thừa của bản cũ), trừ những gì /XD /XF loại ra (không chép, không xoá)
        & robocopy $Dist $AppDir /MIR /R:2 /W:2 /NFL /NDL /NP /NJH /XD (Join-Path $AppDir "weights") /XF "unins*.*" "tu_kiem.log"
        if ($LASTEXITCODE -ge 8) { throw "Chép đè lỗi (robocopy mã $LASTEXITCODE). Đóng Beauty HDR rồi chạy lại CAP_NHAT.bat." }
        if ($UninstKey) {
            try { Set-ItemProperty -LiteralPath $UninstKey -Name "DisplayVersion" -Value $Version -ErrorAction Stop } catch { }
        }
        Step "Tự kiểm bản đã cài, dùng model sẵn có"
        $Log2 = Join-Path $Work "tu_kiem_cap_nhat.log"
        if (Test-Path -LiteralPath $Log2) { Remove-Item -LiteralPath $Log2 -Force }
        $pr = Start-Process -FilePath (Join-Path $AppDir "NetAnh.exe") -ArgumentList "--tu-kiem `"$Log2`" --can-ai" -Wait -PassThru
        if (Test-Path -LiteralPath $Log2) { Get-Content -LiteralPath $Log2 -Encoding UTF8 | ForEach-Object { Write-Host $_ } }
        if ($pr.ExitCode -ne 0) { throw "Đã chép bản $Version nhưng tự kiểm KHÔNG ĐẠT (mã $($pr.ExitCode)), xem $Log2" }
        Write-Host ""
        Write-Host "XONG. Beauty HDR đã lên bản $Version, model giữ nguyên. Mở Beauty HDR từ menu Start như mọi khi." -ForegroundColor Green
        Write-Host "Thư mục _build giữ lại để lần cập nhật sau nhanh; đừng xoá." -ForegroundColor Green
        exit 0
    }

    if ($NoInstaller) {
        $Final = $Dist
    } else {
        Step "6/6 Tạo file cài đặt (Inno Setup)"
        $Iscc = Find-Iscc
        if (-not $Iscc) {
            Install-InnoSetup | Out-Null
            $Iscc = Find-Iscc
            if (-not $Iscc) {
                throw "Đã cài Inno Setup nhưng không thấy ISCC.exe. Đóng cửa sổ này rồi bấm lại file .bat."
            }
        }
        Write-Host "Dùng Inno Setup: $Iscc"
        $Models = if ($NoModels) { "0" } else { "1" }
        if (Test-Path -LiteralPath $Output) { Remove-Item -LiteralPath $Output -Recurse -Force }
        Exec $Iscc @("/Qp", "/O$Output", "/DAppVersion=$Version", "/DFlavor=$Flavor", "/DWithModels=$Models",
                     "/DDistDir=$Dist", "packaging\NetAnh.iss")
        $files = @(Get-ChildItem -LiteralPath $Output -File)
        if ($files.Count -eq 1) {
            $Final = Join-Path $OutDir $files[0].Name
            Copy-Item -LiteralPath $files[0].FullName -Destination $Final -Force
        } else {
            # bản cuda vượt 2 GB: setup.exe kèm các file .bin, phải để chung một thư mục khi cài
            $Final = Join-Path $OutDir "NetAnh-$Version-GPU-NVIDIA"
            New-Item -ItemType Directory -Force -Path $Final | Out-Null
            $files | Copy-Item -Destination $Final -Force
        }
    }

    Write-Host ""
    Write-Host "XONG. File cài đặt: $Final" -ForegroundColor Green
    if (-not $NoInstaller) {
        Write-Host "Bấm đúp file đó để cài Beauty HDR, sau đó mở Beauty HDR từ menu Start. Thư mục _build chỉ chứa file tạm, xoá được." -ForegroundColor Green
    }
    if ($env:CI -ne "true") { Start-Process -FilePath "explorer.exe" -ArgumentList "/select,`"$Final`"" }
    exit 0
} catch {
    Write-Host ""
    Write-Host "LỖI: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Chụp màn hình cửa sổ này gửi lại để sửa. Bấm lại file .bat sẽ làm tiếp, phần đã tải không phải tải lại." -ForegroundColor Red
    exit 1
}
