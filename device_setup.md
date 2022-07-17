Device Setup:
1. install termux from google play
2. change termux repo by following this link - https://www.learntermux.tech/2021/08/termux-repository-under-maintenance.html. Choose A1 repo.
3. install unbuntu on termux: 
Installation steps
    Update termux: apt-get update && apt-get upgrade -y
    Install wget: apt-get install wget -y
    Install proot: apt-get install proot -y
    Install git: apt-get install git -y
    Go to HOME folder: cd ~
    Download script: git clone https://github.com/MFDGaming/ubuntu-in-termux.git
    Go to script folder: cd ubuntu-in-termux
    Give execution permission: chmod +x ubuntu.sh
    Run the script: ./ubuntu.sh -y
    Now just start ubuntu: ./startubuntu.sh

4. get into ubuntu on termux, and use: apt update --> apt install python3 --> apt install pip --> pip3 install torch --> pip3 install numpy.
