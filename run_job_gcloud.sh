if [ $# -eq 1 ]; then
    eval $1
elif [ $1 = "stop-vm" ]; then
    eval $2
    echo "Shutting down VM..."
    sudo shutdown -h now
fi
