# PowerShell Script to simulate a job submission (train.ps1)
<#
    .AUTHOR
        Pallavi Aithal
    .EMAIL
        pallavi.narayan@smail.inf.h-brs.de
    .DATE
        2024-11-14
    .VERSION
        1.0.0
#>

# Set job parameters (you can use variables to adjust them easily)
$jobName = "train"
# $memory = 60GB                    # Memory allocation (not strictly enforced on Windows)
# $cores = 64                       # Number of cores (not enforced; PowerShell uses all available)
$timeLimit = (72 * 60 * 60)       # Time limit in seconds (for display purposes)
$outputFile = "job_output.txt"
$errorFile = "job_error.txt"
$pythonScript = "main.py"
$condaEnvPath = "C:\Users\palla\.conda\envs\sss"

# Activate conda environment
$activateConda = "& "C:\ProgramData\anaconda3\Scripts\activate" $condaEnvPath"
Invoke-Expression $activateConda

# Change to the root directory for the job
Set-Location "D:\FKIE\git_workspace\DeepLabV3Plus-Pytorch"

# Run the Python script and redirect output and error
Start-Process -NoNewWindow -FilePath "python" -ArgumentList $pythonScript `
    -RedirectStandardOutput $outputFile -RedirectStandardError $errorFile
