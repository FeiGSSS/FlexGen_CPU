# flexgen
bash run.sh


# Eval Code
cd FlexGen
conda activate your_env
pip install -e . (or pip install .)
bash eval.sh    
(args can be seen in flexgen.main.add_parser_arguments and eval_utils.___main__.add_evalaute_parser_arguments, the first is model relative args and the second is eval args)