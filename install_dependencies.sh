# Works in Linux Shell on Linux based computers
# Installs the dependencies from the requirements.txt file into the python virtual environment

filename="./requirements.txt"

while read line
do
    pip install $line
done < $filename