# Define the base path
$basePath = Join-Path $PSScriptRoot "..\1_video_material"

# Create the output directory if it doesn't exist
if (-not (Test-Path $basePath)) {
    New-Item -ItemType Directory -Path $basePath | Out-Null
}

# Create a temporary file list for concatenation
$tempList = Join-Path $env:TEMP "concat_list.txt"
@()
1..4 | ForEach-Object {
    "file '$basePath\$_" + ".y4m'"
} | Set-Content -Encoding ASCII $tempList

# Create the master Y4M video
ffmpeg -f concat -safe 0 -i $tempList `
  -vf "fps=30,format=yuv420p" `
  -frames:v 500 `
  -pix_fmt yuv420p `
  -f yuv4mpegpipe "$basePath\video_material.y4m"

# Create scaled YUV420p versions
ffmpeg -i "$basePath\video_material.y4m" -vf "scale=3840:2160" -pix_fmt yuv420p -f yuv4mpegpipe "$basePath\video_material_ultrahd.y4m"
ffmpeg -i "$basePath\video_material.y4m" -vf "scale=1920:1080" -pix_fmt yuv420p -f yuv4mpegpipe "$basePath\video_material_fullhd.y4m"
ffmpeg -i "$basePath\video_material.y4m" -vf "scale=1280:720"  -pix_fmt yuv420p -f yuv4mpegpipe "$basePath\video_material_hd.y4m"

# Create scaled RGB24 raw versions
ffmpeg -i "$basePath\video_material.y4m" -vf "scale=3840:2160" -pix_fmt rgb24 -f rawvideo "$basePath\video_material_ultrahd.rgb"
ffmpeg -i "$basePath\video_material.y4m" -vf "scale=1920:1080" -pix_fmt rgb24 -f rawvideo "$basePath\video_material_fullhd.rgb"
ffmpeg -i "$basePath\video_material.y4m" -vf "scale=1280:720"  -pix_fmt rgb24 -f rawvideo "$basePath\video_material_hd.rgb"

New-Item -ItemType Directory -Force -Path $basePath\jpegs\hd
New-Item -ItemType Directory -Force -Path $basePath\jpegs\fullhd
New-Item -ItemType Directory -Force -Path $basePath\jpegs\ultrahd

# Create jpeg versions with roughly 95 quality
ffmpeg -i "$basePath\video_material_ultrahd.y4m" -q:v 2 -vsync 0 $basePath\jpegs\ultrahd\ultrahd_frame_%04d.jpg 
ffmpeg -i "$basePath\video_material_fullhd.y4m" -q:v 2 -vsync 0 $basePath\jpegs\fullhd\fullhd_frame_%04d.jpg
ffmpeg -i "$basePath\video_material_hd.y4m" -q:v 2 -vsync 0 $basePath\jpegs\hd\hd_frame_%04d.jpg

New-Item -ItemType Directory -Force -Path $basePath\pngs\hd
New-Item -ItemType Directory -Force -Path $basePath\pngs\fullhd
New-Item -ItemType Directory -Force -Path $basePath\pngs\ultrahd

# Create png versions
ffmpeg -i "$basePath\video_material_ultrahd.y4m" -q:v 2 -vsync 0 $basePath\pngs\ultrahd\ultrahd_frame_%04d.png
ffmpeg -i "$basePath\video_material_fullhd.y4m" -q:v 2 -vsync 0 $basePath\pngs\fullhd\fullhd_frame_%04d.png
ffmpeg -i "$basePath\video_material_hd.y4m" -q:v 2 -vsync 0 $basePath\pngs\hd\hd_frame_%04d.png

New-Item -ItemType Directory -Force -Path $basePath\bmps\hd
New-Item -ItemType Directory -Force -Path $basePath\bmps\fullhd
New-Item -ItemType Directory -Force -Path $basePath\bmps\ultrahd

# Create bmp versions
ffmpeg -i "$basePath\video_material_ultrahd.y4m" -q:v 2 -vsync 0 $basePath\bmps\ultrahd\ultrahd_frame_%04d.bmp
ffmpeg -i "$basePath\video_material_fullhd.y4m" -q:v 2 -vsync 0 $basePath\bmps\fullhd\fullhd_frame_%04d.bmp
ffmpeg -i "$basePath\video_material_hd.y4m" -q:v 2 -vsync 0 $basePath\bmps\hd\hd_frame_%04d.bmp

New-Item -ItemType Directory -Force -Path $basePath\tiffs\hd
New-Item -ItemType Directory -Force -Path $basePath\tiffs\fullhd
New-Item -ItemType Directory -Force -Path $basePath\tiffs\ultrahd

# Create tiff versions
ffmpeg -i "$basePath\video_material_ultrahd.y4m" -q:v 2 -vsync 0 $basePath\tiffs\ultrahd\ultrahd_frame_%04d.tiff
ffmpeg -i "$basePath\video_material_fullhd.y4m" -q:v 2 -vsync 0 $basePath\tiffs\fullhd\fullhd_frame_%04d.tiff
ffmpeg -i "$basePath\video_material_hd.y4m" -q:v 2 -vsync 0 $basePath\tiffs\hd\hd_frame_%04d.tiff

# Clean up
Remove-Item $tempList
