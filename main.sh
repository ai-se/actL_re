#!/bin/bash
cd /home/jchen37/actL_re
git pull
mkdir out
mkdir err
rm out/*
rm err/*
for i in p3a p3b p3c osp osp2 flight ground;
do
for k in nsgaii riot;
do
	sbatch -p opteron main.mpi $i $k
	sbatch -p opteron main.mpi $i $k
done;
done;
