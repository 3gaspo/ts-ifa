#!/bin/bash
#SBATCH --job-name=Dynamics
#SBATCH --output=script_outputs/jupyter%j.out
#SBATCH --error=script_outputs/jupyter%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00        
#SBATCH --partition=h100
 
# Charger votre environnement
source .venv/bin/activate
 
# Trouver le nom du nœud sur lequel le job tourne
NODE_NAME=$(hostname)
PORT=8888
 
echo "------------------------------------------------------"
echo "Le serveur Jupyter tourne sur le nœud : $NODE_NAME"
echo "Port utilisé : $PORT"
echo "------------------------------------------------------"
 
# Lancer Jupyter
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0