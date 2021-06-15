script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
ip_list=$(cat $script_dir/ip_list.txt)
docker run --gpus all -d --ipc=host --cap-add=IPC_LOCK -v /sys/class/net/:/sys/class/net/  --device=/dev/ --privileged --network=host -v $main_dir:/root/cogview --name bg-cogview cogview/cuda111_torch181_deepspeed040:base bash -c "/etc/init.d/ssh start && python /root/cogview/env/setup_connection.py $ip_list && sleep 365d" 
