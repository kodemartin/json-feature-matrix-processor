# Generate a sample of files with JSON data
# to test the multi-processing script

# It copies the original file into a folder 'data/'.
# If the folder doesn't exist it is created.
#
# USAGE:
#   $ source generate_data.sh [<number-of-files>]
if [ -z $1 ]; then
    ndata=100
else
    ndata=$1
fi

if [ ! -d "data" ]; then
    mkdir data
fi

for i in `seq 1 $ndata`; do
    cp data.json $(printf "data/data-%04d.json" $i)
done
