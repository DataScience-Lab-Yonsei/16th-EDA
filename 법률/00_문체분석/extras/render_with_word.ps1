param(
    [Parameter(Mandatory = $true)][string]$InputDoc,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [Parameter(Mandatory = $true)][string]$LogPath
)

function Write-Step {
    param([string]$Message)
    Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) $Message" -Encoding UTF8
}

$word = $null
$document = $null
try {
    Write-Step "START"
    $word = New-Object -ComObject Word.Application
    Write-Step "WORD_CREATED"
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.Options.SaveNormalPrompt = $false
    $word.Options.ConfirmConversions = $false
    Write-Step "OPEN_BEGIN"
    $document = $word.Documents.Open($InputDoc, $false, $true, $false)
    Write-Step "OPEN_DONE"
    $document.ExportAsFixedFormat($OutputPdf, 17)
    Write-Step "EXPORT_DONE"
    $document.Close($false)
    $document = $null
    Write-Step "CLOSE_DONE"
}
catch {
    Write-Step "ERROR $($_.Exception.Message)"
    throw
}
finally {
    if ($document -ne $null) {
        try { $document.Close($false) } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() } catch {}
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    Write-Step "END"
}
