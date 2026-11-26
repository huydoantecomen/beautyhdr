# Gói Cập nhật Beauty HDR: chọn thư mục gốc của bản cũ rồi ghi đè mã mới vào đó (không phải chép gói vào đúng chỗ).
# CAP_NHAT.bat / CHUYEN_SANG_GPU.bat trong gói gọi file này.
#
# Thư mục gốc = thư mục có ma-nguon\net_anh\__init__.py (thư mục giải nén bộ Trọn bộ / Windows lần trước, nơi đã chạy
# TAO_FILE_CAI_DAT.bat). Chọn cả thư mục ma-nguon cũng được. Giữ nguyên: ma-nguon\_build (thư viện torch đã tải),
# model (trong thư mục cài đặt và %LOCALAPPDATA%\NetAnh), lịch sử và preset (thư mục người dùng .net_anh).
# Sau khi chép, chạy build_windows.ps1 -CapNhat của thư mục đó như trước (đóng gói lại, chép đè bản đã cài, tự kiểm).
#
# Kiểm thử không cần hộp thoại: -KiemThu (CI Windows chạy, xem .github/workflows/kiem-cap-nhat.yml).
param(
    [string]$Flavor = "",
    [string]$Target = "",          # bỏ qua hộp chọn (dòng lệnh, kiểm thử)
    [string]$NguonMoi = "",        # thư mục mã mới (mặc định: ma-nguon cạnh file .bat)
    [string[]]$TimTrong = @(),     # thư mục tìm bản cũ (mặc định: Desktop, Downloads, Documents, thư mục người dùng, ổ đĩa)
    [switch]$NoBuild,
    [switch]$KiemThu
)
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Pkg = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\BeautyHDR-x.y.z-CapNhat
if (-not $NguonMoi) { $NguonMoi = Join-Path $Pkg "ma-nguon" }
$SaveFile = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "NetAnh\thu_muc_nguon.txt" } else { "" }

function Get-AppRoot([string]$p) {
    # Thư mục gốc hợp lệ (chứa ma-nguon) từ đường dẫn người dùng chọn; $null nếu không phải thư mục Beauty HDR.
    if (-not $p) { return $null }
    $p = $p.TrimEnd('\', '/')
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { return $null }
    if (Test-Path -LiteralPath (Join-Path $p "ma-nguon\net_anh\__init__.py")) { return $p }
    if ((Split-Path -Leaf $p) -ieq "ma-nguon" -and (Test-Path -LiteralPath (Join-Path $p "net_anh\__init__.py"))) {
        return (Split-Path -Parent $p)
    }
    return $null
}

function Get-Ver([string]$src) {
    $f = Join-Path $src "net_anh\__init__.py"
    if (-not (Test-Path -LiteralPath $f)) { return "?" }
    $m = Select-String -LiteralPath $f -Pattern '__version__ = "([^"]+)"'
    if ($m) { return $m.Matches[0].Groups[1].Value }
    return "?"
}

function Test-SamePath([string]$a, [string]$b) {
    try { return ([IO.Path]::GetFullPath($a).TrimEnd('\') -ieq [IO.Path]::GetFullPath($b).TrimEnd('\')) } catch { return $false }
}

function Find-Candidates([string[]]$dirs) {
    # Bản cũ trên máy: đường dẫn đã lưu ở lần cập nhật / build trước, rồi tìm thư mục ma-nguon (sâu tối đa 3 cấp).
    $out = New-Object System.Collections.Generic.List[string]
    if ($SaveFile -and (Test-Path -LiteralPath $SaveFile)) {
        $r = Get-AppRoot ((Get-Content -LiteralPath $SaveFile -Raw -Encoding UTF8).Trim())
        if ($r) { $out.Add($r) }
    }
    if (-not $dirs -or $dirs.Count -eq 0) {
        $dirs = @([Environment]::GetFolderPath("Desktop"), (Join-Path $env:USERPROFILE "Downloads"),
                  [Environment]::GetFolderPath("MyDocuments"), $env:USERPROFILE)
        $dirs += @(Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue | Where-Object { $_.Used -gt 0 } |
                   ForEach-Object { $_.Root })
    }
    foreach ($d in $dirs | Where-Object { $_ } | Select-Object -Unique) {
        if (-not (Test-Path -LiteralPath $d)) { continue }
        $depth = if ($d.Length -le 3) { 2 } else { 3 }   # gốc ổ đĩa: tìm nông hơn cho nhanh
        Get-ChildItem -LiteralPath $d -Directory -Recurse -Depth $depth -Filter "ma-nguon" -ErrorAction SilentlyContinue |
            ForEach-Object {
                $r = Get-AppRoot $_.FullName
                if ($r -and -not (Test-SamePath $r $Pkg) -and -not ($out | Where-Object { Test-SamePath $_ $r })) { $out.Add($r) }
            }
    }
    # ưu tiên: đã có _build (thư viện đã tải, cập nhật nhanh), mới dùng gần đây nhất
    return @($out | Sort-Object -Property @{ Expression = { Test-Path -LiteralPath (Join-Path $_ "ma-nguon\_build") }; Descending = $true },
                                          @{ Expression = { (Get-Item -LiteralPath (Join-Path $_ "ma-nguon")).LastWriteTime }; Descending = $true })
}

function Copy-Update([string]$root) {
    # Ghi đè mã mới vào root\ma-nguon. Không đụng _build (torch), môi trường Python, file tạm; chép kèm .bat và hướng dẫn.
    $dst = Join-Path $root "ma-nguon"
    if (Test-SamePath $NguonMoi $dst) { return }   # gói đã được chép vào đúng chỗ (cách cũ): không cần chép
    & robocopy $NguonMoi $dst /E /R:1 /W:1 /NFL /NDL /NP /NJH /NJS /XD "_build" ".venv" "venv" "__pycache__" ".git" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Chép mã mới lỗi (robocopy mã $LASTEXITCODE). Kiểm tra quyền ghi vào $dst." }
    foreach ($f in @("CAP_NHAT.bat", "CHUYEN_SANG_GPU.bat", "HUONG_DAN_CAP_NHAT.txt")) {
        $s = Join-Path $Pkg $f
        if ((Test-Path -LiteralPath $s) -and -not (Test-SamePath $Pkg $root)) { Copy-Item -LiteralPath $s -Destination $root -Force }
    }
}

function Save-Root([string]$root) {
    if (-not $SaveFile) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SaveFile) | Out-Null
    Set-Content -LiteralPath $SaveFile -Value $root -Encoding UTF8
}

function Choose-Root {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.Application]::EnableVisualStyles()
    $newV = Get-Ver $NguonMoi
    Write-Host "Đang tìm bản Beauty HDR cũ trên máy…"
    $cands = Find-Candidates $TimTrong
    $first = $cands | Select-Object -First 1
    if ($first) {
        $oldV = Get-Ver (Join-Path $first "ma-nguon")
        $fast = if (Test-Path -LiteralPath (Join-Path $first "ma-nguon\_build")) { "Đã có thư viện (torch), cập nhật nhanh." }
                else { "Thư mục này chưa có _build: phải tải lại thư viện (lâu hơn), model vẫn không tải lại." }
        $more = if ($cands.Count -gt 1) { "`n`nCòn $($cands.Count - 1) thư mục khác; bấm No để tự chọn." } else { "" }
        $ans = [System.Windows.Forms.MessageBox]::Show(
            "Tìm thấy Beauty HDR $oldV ở:`n`n$first`n`n$fast`n`nCập nhật thư mục này lên $newV?$more`n`n" +
            "Yes: cập nhật thư mục này · No: chọn thư mục khác · Cancel: thoát",
            "Cập nhật Beauty HDR $newV", "YesNoCancel", "Question")
        if ($ans -eq "Yes") { return $first }
        if ($ans -eq "Cancel") { return $null }
    }
    while ($true) {
        $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
        $dlg.Description = "Chọn thư mục gốc của Beauty HDR bản cũ (thư mục có thư mục ma-nguon bên trong, nơi lần trước bạn " +
                           "đã giải nén và bấm TAO_FILE_CAI_DAT.bat hoặc CAP_NHAT.bat)."
        $dlg.ShowNewFolderButton = $false
        if ($first) { $dlg.SelectedPath = $first }
        if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
        $r = Get-AppRoot $dlg.SelectedPath
        if ($r -and (Test-SamePath $r $Pkg)) {
            [System.Windows.Forms.MessageBox]::Show("Đây là chính gói cập nhật vừa giải nén. Hãy chọn thư mục của bản cũ.",
                "Chọn lại", "OK", "Warning") | Out-Null
            continue
        }
        if ($r) { return $r }
        [System.Windows.Forms.MessageBox]::Show(
            "Thư mục này không phải thư mục Beauty HDR (không thấy ma-nguon\net_anh bên trong):`n`n$($dlg.SelectedPath)`n`n" +
            "Hãy chọn thư mục chứa thư mục ma-nguon của bản cũ.", "Chọn lại", "OK", "Warning") | Out-Null
        $first = $dlg.SelectedPath
    }
}

function Invoke-KiemThu {
    # Thư mục giả: bản cũ 0.14.2 có _build và file thư viện, gói mới; kiểm nhận thư mục, tìm, chép đè, giữ _build.
    $t = Join-Path ([IO.Path]::GetTempPath()) ("kiem_cap_nhat_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    $old = Join-Path $t "nét ảnh\BeautyHDR-0.14.2-Windows"
    New-Item -ItemType Directory -Force -Path (Join-Path $old "ma-nguon\net_anh"), (Join-Path $old "ma-nguon\_build\venv-cuda\Lib") | Out-Null
    Set-Content -LiteralPath (Join-Path $old "ma-nguon\net_anh\__init__.py") -Value '__version__ = "0.14.2"' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $old "ma-nguon\_build\venv-cuda\Lib\torch.txt") -Value "torch cũ" -Encoding UTF8
    $bad = Join-Path $t "khac"
    New-Item -ItemType Directory -Force -Path $bad | Out-Null
    $script:fail = 0
    function Check($ok, $what) { if ($ok) { Write-Host "ĐẠT  $what" } else { Write-Host "LỖI  $what"; $script:fail++ } }
    Check ((Get-AppRoot $old) -eq $old) "nhận thư mục gốc"
    Check ((Get-AppRoot (Join-Path $old "ma-nguon")) -eq $old) "chọn thư mục ma-nguon cũng được"
    Check ($null -eq (Get-AppRoot $bad)) "từ chối thư mục không phải Beauty HDR"
    Check ($null -eq (Get-AppRoot (Join-Path $t "khong-co"))) "từ chối thư mục không tồn tại"
    $script:SaveFile = Join-Path $t "luu\thu_muc_nguon.txt"
    $c = @(Find-Candidates @($t))
    Check ($c.Count -eq 1 -and (Test-SamePath $c[0] $old)) "tìm thấy bản cũ ($($c -join '; '))"
    Copy-Update $old
    Check ((Get-Ver (Join-Path $old "ma-nguon")) -eq (Get-Ver $NguonMoi)) "đã ghi đè lên bản $(Get-Ver $NguonMoi)"
    Check (Test-Path -LiteralPath (Join-Path $old "ma-nguon\_build\venv-cuda\Lib\torch.txt")) "giữ nguyên _build (torch)"
    Check (Test-Path -LiteralPath (Join-Path $old "ma-nguon\packaging\build_windows.ps1")) "có build_windows.ps1 mới"
    Save-Root $old
    Check ((Get-Content -LiteralPath $script:SaveFile -Raw -Encoding UTF8).Trim() -eq $old) "ghi nhớ thư mục cho lần sau"
    Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue
    if ($script:fail) { Write-Host "$($script:fail) kiểm tra LỖI"; exit 1 }
    Write-Host "Tất cả kiểm tra ĐẠT"; exit 0
}

if ($KiemThu) { Invoke-KiemThu }

try {
    if (-not (Test-Path -LiteralPath (Join-Path $NguonMoi "net_anh\__init__.py"))) {
        throw "Không thấy mã mới ($NguonMoi). Hãy giải nén CẢ gói cập nhật rồi bấm đúp CAP_NHAT.bat trong đó."
    }
    $root = $null
    if ($Target) {
        $root = Get-AppRoot $Target
        if (-not $root) { throw "Không phải thư mục Beauty HDR: $Target" }
    } elseif (Test-Path -LiteralPath (Join-Path $NguonMoi "_build")) {
        $root = $Pkg   # cách cũ: gói đã được chép vào thư mục bản cũ (có _build) → cập nhật ngay tại đây
        Write-Host "Gói cập nhật đang nằm trong thư mục bản cũ: cập nhật ngay tại đây."
    } else {
        $root = Choose-Root
        if (-not $root) { Write-Host "Đã huỷ, chưa thay đổi gì."; exit 1 }
    }
    Write-Host "Cập nhật: $root (bản $(Get-Ver (Join-Path $root 'ma-nguon')) → $(Get-Ver $NguonMoi))" -ForegroundColor Green
    Write-Host "Giữ nguyên: thư viện đã tải (ma-nguon\_build), model, lịch sử và preset."
    Copy-Update $root
    Save-Root $root
    if ($NoBuild) { exit 0 }
    $build = Join-Path $root "ma-nguon\packaging\build_windows.ps1"
    $args2 = @("-CapNhat")
    if ($Flavor) { $args2 += @("-Flavor", $Flavor) }
    & $build @args2
    exit $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "LỖI: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Chưa chép gì vào bản đã cài; bản cũ vẫn dùng được." -ForegroundColor Yellow
    exit 1
}
