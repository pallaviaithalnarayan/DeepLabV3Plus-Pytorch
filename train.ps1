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
$outputFile = "job_output.out"
$errorFile = "job_error.err"
$pythonScript = "main.py"
$condaEnvPath = "C:\Users\palla\.conda\envs\sss"
$jobArgs = "--model deeplabv3plus_mobilenet --num_classes 7 --dataset mydata --gpu_id 0 --lr 0.001 --crop_size 512 --batch_size 16 --output_stride 16 --data_root ./datasets/data_uavid/ --save_val_results --separable_conv --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"

pip install visdom
python -c "import visdom"
# Activate conda environment
$activateConda = "& 'C:\ProgramData\anaconda3\Scripts\activate' $condaEnvPath"
Invoke-Expression $activateConda

# Change to the root directory for the job
Set-Location "D:\FKIE\git_workspace\DeepLabV3Plus-Pytorch"

# Run the Python script and redirect output and error
# Start-Process -NoNewWindow -FilePath "python" -ArgumentList $pythonScript `
#     > $outputFile 2> $errorFile

# Run the Python script and redirect output and error synchronously
# python $pythonScript $jobArgs > $outputFile 2> $errorFile

Start-Process -FilePath "python" -ArgumentList "$pythonScript $jobArgs" -NoNewWindow -RedirectStandardOutput $outputFile -RedirectStandardError $errorFile -Wait

