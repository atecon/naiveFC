#!/bin/bash
set -e

DIR=$(dirname $(realpath "$0")) 	# locate folder where this sh-script is located in

PROJECT="naiveFC"

SCRIPT1="./tests/run_tests.inp"
SCRIPT2="./tests/run_tests_static_case.inp"
#SCRIPT3="./tests/run_tests_moving_window_case.inp"

cd $DIR
echo "Switched to ${DIR}"


gretlcli -b -e -q ${SCRIPT1}
if [ $? -eq 0 ]
then
  echo "Success: Tests for script ${SCRIPT1} passed."
else
  echo "Failure: Tests for script ${SCRIPT1} not passed."
fi


gretlcli -b -e -q ${SCRIPT2}
if [ $? -eq 0 ]
then
  echo "Success: Tests for script ${SCRIPT2} passed."
else
  echo "Failure: Tests for script ${SCRIPT2} not passed."
fi


#gretlcli -b -e -q ${SCRIPT3}
#if [ $? -eq 0 ]
#then
 # echo "Success: Tests for script ${SCRIPT3} passed."
#else
#  echo "Failure: Tests for script ${SCRIPT3} not passed." >&2
#fi




# if [ $? -eq 0 ]
# then
#   echo "Success: All tests passed for '${PROJECT}'."
#   exit 0
# else
#   echo "Failure: Tests not passed for '${PROJECT}'."
#   exit 1
# fi

