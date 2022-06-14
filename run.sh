# SAT-NoiLIn with symmetric-flipping noisy labels using ResNet-18
python SAT-NoiLIn.py --dataset='cifar10' --noise_type='symmetric'
python SAT-NoiLIn.py --dataset='cifar100' --noise_type='symmetric'
python SAT-NoiLIn.py --dataset='svhn' --noise_type='symmetric'

# SAT-NoiLIn with symmetric-flipping noisy labels using WRN-32-10
python SAT-NoiLIn.py --dataset='cifar10' --noise_type='symmetric' --net="WRN_madry" 

# TRADES-NoiLIn with symmetric-flipping noisy labels using ResNet-18
python TRADES-NoiLIn.py --dataset='cifar10' --noise_type='symmetric'
python TRADES-NoiLIn.py --dataset='cifar100' --noise_type='symmetric'
python TRADES-NoiLIn.py --dataset='svhn' --noise_type='symmetric'

# TRADES-NoiLIn with symmetric-flipping noisy labels using WRN-34-10
python TRADES-NoiLIn.py --dataset='cifar10' --noise_type='symmetric' --net="WRN" 

# TRADES-AWP-NoiLIn with symmetric-flipping noisy labels using WRN-34-10
cd TRADES-AWP-NoiLIn
python TRADES-AWP-NoiLIn.py

# SAT-NoiLIn with extra training data using WRN-28-10 
cd NoiLIn_ExtraData
python SAT-NoiLIn-ExtraData.py --gpu='0,1,2,3' --aux_data_filename='ti_500K_pseudo_labeled.pickle'

# Obtain the learning curve of natural and robust accuracy
python eval.py --all_epoch --start_epoch=1 --end_epoch=120 --model_dir='model_dir'

# Obtain natural and robust accuracy of a given model 
python eval.py --model_dir='model_dir' --pt_name='model_pt_name'