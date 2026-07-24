param(
    [string]$Mpt = "PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt",
    [int]$Cycle = 1,
    [ValidateSet("Ewe", "Ece")]
    [string]$Control = "Ece",
    [double]$Threshold = 1.0,
    [string]$Circuit = "R0-L0-p(R1,CPE1)"
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

uv run python main.py $Mpt --cycle $Cycle --control $Control --threshold $Threshold --circuit $Circuit
