$destinationFolder = Join-Path -Path $PSScriptRoot -ChildPath "..\1_video_material"

# Ensure the destination directory exists
if (-not (Test-Path -Path $destinationFolder)) {
    New-Item -ItemType Directory -Path $destinationFolder | Out-Null
}

$urls = @(
    "https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m",
    "https://media.xiph.org/video/derf/ElFuente/Netflix_Narrator_4096x2160_60fps_10bit_420.y4m",
    "https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m",
    "https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m"
)

for ($i = 0; $i -lt $urls.Count; $i++) {
    $outputFile = Join-Path -Path $destinationFolder -ChildPath "$($i + 1).y4m"
    Invoke-WebRequest -Uri $urls[$i] -OutFile $outputFile
}
