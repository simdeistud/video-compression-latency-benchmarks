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

foreach ($url in $urls) {
    $filename = Split-Path $url -Leaf
    $outputPath = Join-Path -Path $destinationFolder -ChildPath $filename
    Invoke-WebRequest -Uri $url -OutFile $outputPath
}
