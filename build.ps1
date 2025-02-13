[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseDeclaredVarsMoreThanAssignments', '')]
param (
    [switch]$r
)

$Green = @{
    ForegroundColor = 'Green'
}
$Yellow = @{
    ForegroundColor = 'Yellow'
}

$LocalBaseImage = "remote-client-server-base:latest"
$RegistryBaseImage = "docker.semoss.org/genai/remote-client-server-base:latest"

function Initialize-BaseImage {
    Write-Host "Checking if base image needs to be built..." @Yellow
    
    $existingImage = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -eq $LocalBaseImage }
    
    if ($existingImage) {
        Write-Host "Base image already exists locally" @Green
        return $true
    }
    else {
        Write-Host "Base image not found locally, building..." @Yellow
        $buildResult = docker build -f Dockerfile.base -t $LocalBaseImage .
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Base image build failed!" @Yellow
            return $false
        }
        
        Write-Host "Base image built successfully!" @Green
        return $true
    }
}

function Initialize-ServerImage {
    param (
        [string]$BaseImage
    )
    
    Write-Host "Building server image with base: $BaseImage" @Green
    $buildResult = docker build --build-arg BASE_IMAGE=$BaseImage -t remote-client-server:latest .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Server image build failed!" @Yellow
        exit 1
    }
    
    Write-Host "Server image built successfully!" @Green
}

if ($r) {
    Write-Host "Using registry base image..." @Yellow
    Initialize-ServerImage -BaseImage $RegistryBaseImage
}
else {
    Write-Host "Using local base image..." @Yellow
    $baseResult = Initialize-BaseImage
    if ($baseResult) {
        Initialize-ServerImage -BaseImage $LocalBaseImage
    }
    else {
        exit 1
    }
}