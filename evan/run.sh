nohup python main.py --env ant-expert-v2 --max_pretrain_iters 10 --exp_name ant_expert_pretrain_iters100 > nohup.log 2>&1 & echo $! > nohup.pid
nohup python main.py --env agent_sac --max_pretrain_iters 10 --exp_name finrl_pretrain_iters100 > nohup.log 2>&1 & echo $! > nohup.pid
nohup python main.py --env finrl-dt_td3 --max_online_iters 50 --exp_name finrl-dt_td3 > log/train_finrl-dt_td3.log 2>&1 & echo $! > log/train_finrl-dt_td3.pid

# 使用finrl-dt-td3製作的trajectories 裡面有1000份trajectories
nohup python main.py --env finrl-dt_td3-1000 --max_online_iters 50 --exp_name finrl-dt_td3_1000 > log/train_finrl-dt_td3-1000.log 2>&1 & echo $! > log/train_finrl-dt_td3-1000.pid

# 使用finrl-dt-td3製作的trajectories 裡面有100份trajectories
nohup python main.py --env train_data_offline_td3_100 --max_online_iters 50 --exp_name train_data_offline_td3_100 > log/train_data_offline_td3_100.log 2>&1 & echo $! > log/train_data_offline_td3_100.pid

# 使用finrl-dt-a2c製作的trajectories 裡面有100份trajectories
nohup python main.py --env finrl-dt_train_offline_a2c_100 --max_online_iters 50 --exp_name train_offline_a2c_100 > log/train_offline_a2c_100.log 2>&1 & echo $! > log/train_offline_a2c_100.pid
