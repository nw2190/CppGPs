#!/bin/bash
clear;
printf "\n"
echo "--------------------------------"
echo "|         1D EXAMPLE           |"
echo "--------------------------------"
printf "\n--------------- CppGPs ---------------\n"
./1D_example
printf "\n------------ SciKit Learn ------------\n\n"
python Comparison_with_SciKit_Learn.py
printf "\n-------------- GPyTorch ---------------\n"
python Comparison_with_GPyTorch.py

printf "\n\n\n"
echo "--------------------------------"
echo "|         2D EXAMPLE           |"
echo "--------------------------------"
printf "\n--------------- CppGPs ---------------\n"
./2D_example
printf "\n------------ SciKit Learn ------------\n\n"
python Comparison_with_SciKit_Learn.py
printf "\n-------------- GPyTorch ---------------\n"
python Comparison_with_GPyTorch.py


printf "\n\n\n"
echo "--------------------------------"
echo "|        2D MULTIMODAL         |"
echo "--------------------------------"
printf "\n--------------- CppGPs ---------------\n"
./2D_multimodal
printf "\n------------ SciKit Learn ------------\n\n"
python Comparison_with_SciKit_Learn.py
printf "\n-------------- GPyTorch ---------------\n"
python Comparison_with_GPyTorch.py
