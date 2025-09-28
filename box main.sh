#！/bin/bash
for i in {7..14}
do
	python3 qua_box.py $((2 ** i))
done