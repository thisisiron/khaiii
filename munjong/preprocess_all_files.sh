#!/bin/bash

echo "Input Path: $1"

# 첫 번째로 실행
#for i in $1/*.txt; do ./recover_raw_morph_mismatch.py --input ${i} --output $1_mismatch/$(basename ${i}); done
#echo "Complete fix_final_symbol"

#for i in $1_mismatch/*.txt; do ./convert_jamo_to_compat.py --input ${i} --output $1_convert_jamo/$(basename ${i});  done
#echo "Complete convert_jamo_to_compat"

#for i in $1_convert_jamo/*.txt; do ./recover_english_case.py --input ${i} --output $1_re_eng/$(basename ${i}); done
#echo "Complete recover_english_case"

#for i in $1_re_eng/*.txt; do ./fix_final_symbol_error.py --input ${i} --output $1_final_symbol/$(basename ${i}); done
#echo "Complete fix_final_symbol_error"

#for i in $1_final_symbol/*.txt; do ./recover_wide_quotation.py --input ${i} --output $1_re_quotation/$(basename ${i}); done
#echo "Complete recover_wide_quotation"

#for i in $1_re_quotation/*.txt; do ./remove_sejong_period_error.py --input ${i} --output $1_rm_period/$(basename ${i}); done
#echo "Complete remove_sejong_period_error"

for i in $1_rm_period/*.txt; do ./detect_sejong_period_error.py --input ${i} --output $1_detect_period/$(basename ${i}); done
echo "Complete detect_sejong_period_error"

