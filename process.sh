echo "processing $1"
file=$1
python process-auto.py --input_file_name $1 --out_file_name "$1.xlsx"
