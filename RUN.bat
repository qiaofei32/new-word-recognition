@echo off
REM python new-word-recognition.py -m get_words -i corpus/all-comment.txt -t 100 -n 10000000
python new-word-recognition.py --method get_words --input corpus/data.txt --tf 100 --lines 10000000
pause