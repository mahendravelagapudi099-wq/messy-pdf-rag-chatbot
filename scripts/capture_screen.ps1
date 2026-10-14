<#
.SYNOPSIS
    Captures a high-resolution screenshot of the active desktop/browser and saves it for GitHub proof.

.PARAMETER OutputFile
    Path where the screenshot PNG will be saved. Default is 'docs/screenshots/live_app_proof.png'.

.PARAMETER DelaySeconds
    Seconds to wait before snapping the screenshot, allowing time to focus or switch tabs. Default is 2.

.EXAMPLE
    .\scripts\capture_screen.ps1 -OutputFile "docs/screenshots/chat_grounding.png"
#>

param(
    [string]$OutputFile = "docs/screenshots/live_app_proof.png",
    [int]$DelaySeconds = 2
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if ($DelaySeconds -gt 0) {
    Write-Host "Capturing screen in $DelaySeconds second(s)... Please ensure http://localhost:8501 is visible on screen." -ForegroundColor Cyan
    Start-Sleep -Seconds $DelaySeconds
}

$screen = [System.Windows.Forms.Screen]::PrimaryScreen
$bounds = $screen.Bounds
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)

try {
    $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    
    $parentDir = Split-Path -Parent $OutputFile
    if ($parentDir -and -not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    
    $bitmap.Save($OutputFile, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Host "SUCCESS: Live screenshot saved to '$OutputFile'" -ForegroundColor Green
}
catch {
    Write-Error "Failed to capture screenshot: $_"
}
finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
